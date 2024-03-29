import datetime
import toml

def update_version_in_pyproject(file_path: str):

    with open(file_path, 'r') as file:
        pyproject_data = toml.load(file)
    now = datetime.datetime.now()
    new_version = f"999.1.13-dev-{now.strftime('%Y%m%d-%H%M%S')}"

    pyproject_data['project']['version'] = new_version

    with open(file_path, 'w') as file:
        toml.dump(pyproject_data, file)

    print(f"Version updated to {new_version}")

file_path = "pyproject.toml"
update_version_in_pyproject(file_path)
