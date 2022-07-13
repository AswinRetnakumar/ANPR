import os
import re
import shutil

file_name_pattern = '[A-Z0-9]{1,10}_?\d?$'

def check_consistency(folder):
    folder_path = os.path.join(folder)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)

        for file in files:
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                file_ext = file.split('.')
                if file_ext[-1] in ['jpg', 'png']:
                    if re.match(file_name_pattern, file_ext[0]):
                        continue
                    else:
                        print("Failed: ", file_ext[0])
                        shutil.move(file_path, os.path.join('skipped_images'))
                else:
                    continue
    else:
        raise ValueError

if __name__ == "__main__":
    check_consistency('ANPR_indian_plates_labelled')