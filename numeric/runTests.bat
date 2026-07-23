@echo  off

for %%f in (*.py) do (
    echo Running %%f
    python "%%f"
)