cp ../README.md ../LICENSE .
rm -rf dist
python update_version.py
git add pyproject.toml
python3 -m build
#python3 -m twine upload dist/*
rm /monster/data/pkg/sglang*.whl
rm -rf README.md LICENSE
mv dist/*.whl /monster/data/pkg/
