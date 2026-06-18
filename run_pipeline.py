# run_pipeline.py
import os
import sys
import glob
import shutil
import zipfile
import tempfile
import importlib
import argparse
import asyncio
import subprocess

# Setup command-line argument parser to allow configuration customization
parser = argparse.ArgumentParser(description="Run Spelling Alignment Pipeline via GitHub Actions")
parser.add_argument("--project", type=str, default="project_2", help="Directory name of the project config")
parser.add_argument("--config", type=str, default="pipeline_config.json", help="Configuration file name")
parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite", help="Gemini model to use")
parser.add_argument("--phase", type=int, default=1, help="Current execution phase")
parser.add_argument("--total-phases", type=int, default=1, help="Total execution phases")
parser.add_argument("--concurrency", type=int, default=2, help="Concurrent request limit")
args = parser.parse_args()

# --- Configuration & Path Settings ---
WORKSPACE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "pipeline_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define configurations to pass to the Orchestrator
pipeline_config = {
    "PIPELINE_CONFIG_FILENAME": args.config,
    "SELECTED_MODEL": args.model,
    "MOUNT_GDRIVE": False,
    "GDRIVE_ZIP_CORE_PATH": os.path.join(WORKSPACE_DIR, "pipeline_core.zip"),
    "GDRIVE_ZIP_PROMPTS_PATH": os.path.join(WORKSPACE_DIR, "prompts_package.zip"),
    "GDRIVE_ZIP_ASSETS_PATH": os.path.join(WORKSPACE_DIR, "assets.zip"),
    "GDRIVE_OUTPUT_DIR": OUTPUT_DIR,
    "LOCAL_WORKDIR": os.path.join(WORKSPACE_DIR, "local_prompts_workspace"),
    "LOCAL_RESPONSES_DIR": os.path.join(WORKSPACE_DIR, "local_responses"),
    "CHECKPOINT_INTERVAL": 5,
    "PERIODIC_CHECKPOINT_INTERVAL": 300,
    "MAX_RETRIES": 3,
    "AUTO_TUNE_CONCURRENCY": False,
    "CONCURRENT_REQUEST_SIZE": args.concurrency,
    "KEY_RPM_LIMIT": 2,
    "KEY_DAILY_LIMIT": 500,
    "KEY_TPM_LIMIT": 250000,
    "MAX_RESPONSE_TOKENS": 65000,
    "GET_MIDDLE_STEPS_BACKUP": True,
    "RETRY_FAILED_ITEMS": True,
    "THINKING_BUDGET": -1,
    "THINKING_LEVEL": "High",
    "CURRENT_PHASE": args.phase,
    "TOTAL_PHASES": args.total_phases,
}

# Load API keys from environment variable (set via GitHub Secrets)
api_keys_env = os.environ.get("GEMINI_API_KEYS", "")
if api_keys_env:
    pipeline_config["API_KEYS"] = [k.strip() for k in api_keys_env.split(",") if k.strip()]
else:
    print("WARNING: No GEMINI_API_KEYS environment variable found. Checking fallback keys.")
    pipeline_config["API_KEYS"] = []

# Path Resolution Helper
def find_file(configured_path, fallback_dir):
    if os.path.exists(configured_path):
        return configured_path
    filename = os.path.basename(configured_path)
    fallback_path = os.path.join(fallback_dir, filename) if fallback_dir else None
    if fallback_path and os.path.exists(fallback_path):
        return fallback_path
    return None

def encrypt_output_zips(directory, password):
    """
    Locates all generated .zip archives in the output folder, extracts them,
    and re-packages them with AES password protection using the system 'zip' tool.
    """
    if not password:
        print("ℹ️ No ZIP_PASSWORD secret provided. Skipping password-protection encryption step.")
        return

    zip_files = glob.glob(os.path.join(directory, "*.zip"))
    book_zips = [f for f in zip_files if "responses_checkpoint" not in os.path.basename(f)]

    if not book_zips:
        print("ℹ️ No book group ZIP files found in the output directory to encrypt.")
        return

    print(f"🔒 Found {len(book_zips)} book ZIP file(s). Beginning password encryption...")
    for zip_path in book_zips:
        filename = os.path.basename(zip_path)
        with tempfile.TemporaryDirectory() as temp_extract_dir:
            try:
                # Extract unencrypted zip contents
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(temp_extract_dir)
                
                # Delete unencrypted zip
                os.remove(zip_path)
                
                # Re-archive with encryption via CLI tool
                subprocess.run(
                    ["zip", "-q", "-P", password, "-r", zip_path, "."],
                    cwd=temp_extract_dir,
                    check=True
                )
                print(f"✅ Successfully encrypted book: '{filename}'")
            except Exception as e:
                print(f"❌ Failed to encrypt '{filename}': {e}")

async def main():
    local_workdir = pipeline_config["LOCAL_WORKDIR"]
    os.makedirs(local_workdir, exist_ok=True)

    core_path = find_file(pipeline_config["GDRIVE_ZIP_CORE_PATH"], pipeline_config["GDRIVE_OUTPUT_DIR"])
    prompts_path = find_file(pipeline_config["GDRIVE_ZIP_PROMPTS_PATH"], pipeline_config["GDRIVE_OUTPUT_DIR"])
    assets_path = find_file(pipeline_config["GDRIVE_ZIP_ASSETS_PATH"], pipeline_config["GDRIVE_OUTPUT_DIR"])

    # 1. Unpack Core Engine
    if core_path:
        print(f"📦 Processing core engine package from: '{core_path}'...")
        dest_core_dir = os.path.join(local_workdir, "pipeline_core")
        os.makedirs(dest_core_dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(core_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            contents = os.listdir(temp_dir)
            src_dir = os.path.join(temp_dir, "pipeline_core") if len(contents) == 1 and contents[0] == "pipeline_core" else temp_dir
            for filename in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir, filename), os.path.join(dest_core_dir, filename))
        print(f"✅ Successfully loaded core engine to: '{dest_core_dir}'")
    else:
        print("⚠️ WARNING: Could not find core engine package.")

    # 2. Unpack Prompts / Config Package
    if prompts_path:
        print(f"📦 Processing prompts and configuration package from: '{prompts_path}'...")
        shutil.unpack_archive(prompts_path, local_workdir)
        print("✅ Successfully loaded prompts and configuration.")
    else:
        project_src = os.path.join(WORKSPACE_DIR, args.project)
        if os.path.exists(project_src):
            print(f"ℹ️ Copying prompts from local folder '{project_src}' to '{local_workdir}'...")
            shutil.copytree(project_src, local_workdir, dirs_exist_ok=True)
        else:
            print(f"⚠️ WARNING: Could not find prompts package or direct folder '{args.project}'.")

    # 3. Unpack Raw Source Assets
    if assets_path:
        print(f"📦 Processing assets package from: '{assets_path}'...")
        dest_assets_dir = os.path.join(local_workdir, "assets")
        os.makedirs(dest_assets_dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(assets_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            contents = os.listdir(temp_dir)
            src_dir = os.path.join(temp_dir, "assets") if len(contents) == 1 and contents[0] == "assets" else temp_dir
            for filename in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir, filename), os.path.join(dest_assets_dir, filename))
        print(f"✅ Successfully loaded assets to: '{dest_assets_dir}'")
    else:
        print("⚠️ WARNING: Could not find assets package.")

    # Validate Core Folder
    if not os.path.exists(os.path.join(local_workdir, "pipeline_core")):
        print("\n❌ CRITICAL ERROR: The 'pipeline_core' directory was not found.")
        sys.exit(1)

    if local_workdir not in sys.path:
        sys.path.append(local_workdir)

    importlib.invalidate_caches()
    from pipeline_core.engine import Orchestrator

    try:
        orchestrator = Orchestrator(pipeline_config)
        results = await orchestrator.run()
        print("\n✅ Run execution finished.")
        
        # Run ZIP Encryption post-processor prior to exiting
        zip_password = os.environ.get("ZIP_PASSWORD", "")
        encrypt_output_zips(OUTPUT_DIR, zip_password)
        
    except Exception as e:
        print(f"\n❌ Pipeline halted: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
