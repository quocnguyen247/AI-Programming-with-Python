import zipfile
import os

def zip_folder_recursively(folder_path, output_zip):
    """
    Recursively zips all files and folders in a given directory.
    
    :param folder_path: Path to the root folder to zip.
    :param output_zip: The name or path for the output zip file.
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Full path of the file to add to the zip file
                file_path = os.path.join(root, file)
                # Relative path to retain folder structure within the zip
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
                print(f"Added {file_path} as {arcname}")
    
    print(f"Folder {folder_path} and all subfolders zipped into {output_zip}")

# Example usage
folder_path = 'Cat_Dog_data'  # Replace with the folder path you want to zip
output_zip = 'aipnd-project-part2.zip'  # Name of the resulting zip file
zip_folder_recursively(folder_path, output_zip)
