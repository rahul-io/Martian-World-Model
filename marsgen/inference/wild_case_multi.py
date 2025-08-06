import os
import shutil
import subprocess
import glob

def copy_and_run(pose_file_path, id_name):
    # Copy to 0.txt
    target_dir = "/mnt/workspace/yyy/yyy/ac3d/case/pose_files"
    target_file = os.path.join(target_dir, "0.txt")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy source file to 0.txt
    shutil.copy2(pose_file_path, target_file)
    
    # Copy 0.txt to 1-20.txt
    png_files = glob.glob("/mnt/workspace/yyy/yyy/ac3d/case/real10k_/images/*.png")
    for png in png_files:
        id=os.path.basename(png).split('.')[0]
        dest_file = os.path.join(target_dir, f"{id}.txt")
        try:
            shutil.copy2(target_file, dest_file)
            print(f"Created {dest_file}")
        except Exception as e:
            print(f"Error creating {dest_file}: {e}")
    
    # Run bash script
    script_path = "/mnt/workspace/yyy/yyy/ac3d/inference/run_multi_gpu.sh"
    subprocess.run(f"chmod +x {script_path}", shell=True, check=True)
    
    # Run bash script
    bash_command = f"{script_path} 4000 4000 200 {id_name}"
    try:
        subprocess.run(bash_command, shell=True, check=True)
        print(f"Successfully executed bash command for {id_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing bash command: {e}")

def main():
    # Define directories
    # stride3_dir = "/mnt/workspace/yyy/yyy/ac3d/case/real10k/stride3/pose_files"
    # stride5_dir = "/mnt/workspace/yyy/yyy/ac3d/case/real10k/stride5/pose_files"
    
    # # Process stride3 files
    # for pose_file in glob.glob(os.path.join(stride3_dir, "*.txt")):
    #     id_name = os.path.basename(pose_file).replace('.txt', '')
    #     print(f"\nProcessing stride3 file: {id_name}")
    #     copy_and_run(pose_file, f"s3_{id_name}")
    
    # # Process stride5 files
    # for pose_file in glob.glob(os.path.join(stride5_dir, "*.txt")):
    #     id_name = os.path.basename(pose_file).replace('.txt', '')
    #     print(f"\nProcessing stride5 file: {id_name}")
    #     copy_and_run(pose_file, f"s5_{id_name}")
    
    stride3_dir = "/mnt/workspace/common/Datasets/DL3DV_Dataset/dataset_forcogvideo/dl3dv_10k/good_case_for_test_2s"
    for pose_file in glob.glob(os.path.join(stride3_dir, "*.txt")):
        id_name = os.path.basename(pose_file).replace('.txt', '')
        copy_and_run(pose_file, f"select_cam_stride3-5_{id_name}")

if __name__ == "__main__":
    main()