import os
import shutil

# Set the path to your parent directory
parent_dir = "/Users/sebasdeldi/Development/SD/people_dataset"

# Ensure the path exists
if not os.path.exists(parent_dir):
    print(f"Error: The directory '{parent_dir}' does not exist.")
    exit()

# Get all subfolders inside the parent directory
subfolders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]

for subfolder in subfolders:
    subfolder_path = os.path.join(parent_dir, subfolder)
    files = sorted(os.listdir(subfolder_path))  # Sort files for consistency
    counter = 1

    for file in files:
        file_path = os.path.join(subfolder_path, file)

        # Ensure it's a file and not a directory
        if os.path.isfile(file_path):
            # Extract the file extension
            ext = os.path.splitext(file)[1]
            
            # Replace spaces in subfolder names with underscores in the new filename (optional)
            sanitized_folder_name = subfolder.replace(" ", "_")
            
            # Create the new filename
            new_filename = f"{sanitized_folder_name}_{counter}{ext}"
            new_file_path = os.path.join(parent_dir, new_filename)

            # Move the file
            shutil.move(file_path, new_file_path)
            print(f"Moved: {file_path} → {new_file_path}")

            counter += 1  # Increment counter

print("All images moved successfully!")