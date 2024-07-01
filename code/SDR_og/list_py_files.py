import os

def list_py_files(directory):
    py_files = [os.path.join(root, file) 
                for root, dirs, files in os.walk(directory) 
                for file in files if file.endswith('.py')]
    return py_files

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def save_to_txt(directory, output_file):
    py_files = list_py_files(directory)
    with open(output_file, 'w') as out_file:
        for py_file in py_files:
            out_file.write(f"Contents of {py_file}:\n")
            out_file.write(read_file(py_file))
            out_file.write("\n" + "="*80 + "\n\n")

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    output_file = input("Enter the output file name (with .txt extension): ")
    save_to_txt(directory, output_file)
    print(f"Contents of .py files have been saved to {output_file}")

