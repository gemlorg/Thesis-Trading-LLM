# Project Guidelines and Rules

## Task Completion Algorithm

1. **Create a New Branch**:

   - Create a separate branch in the repository for your task.

2. **Organize Your Solution**:

   - Create a corresponding folder for your solution.
   - Follow the kebab-case naming convention for folder names (e.g., `linear-regression`).
   - Create the following project structure inside your folder:

   ```
   project-root/
   └── your-task-folder/
       ├── data/
       ├── .gitignore
       ├── Pipfile
       ├── Pipfile.lock (add to .gitignore)
       ├── README.md
       ├── results/
       │   └── SUMMARY.md
       └── src/
   ```

3. **Create a Pipfile**:

   - Create a Pipfile in your project directory to manage dependencies.

4. **Implement the Task**:

   - Write and implement your task following the coding principles.

5. **Report Your Work**:

   - Add a summary or report inside the "results" folder to describe your work (more details in "Reporting Principles").

6. **Consult with the Team**:
   - Before merging your changes with the main branch, consult with your team to ensure compatibility.

## Code Writing Principles

- Follow snake_case naming convention for variables and functions.
- Use [pipenv](https://packaging.python.org/tutorials/managing-dependencies/) to manage dependencies.
- Install all dependencies locally inside your project directory. (you have to have a blank file named Pipfile inside your project directroy)
- Add Pipfile.lock to .gitignore.
- Write code and comments in English.
- Format your code using [black](https://black.readthedocs.io/en/stable/).
- Comments should explain "why" not "how".
- Consider using a spell checker for code quality.

## Communication Principles

- The team should meet at least one day before the deadline with all tasks completed to discuss the results.

## Reporting Principles

- Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages (mostly use feat, chore, fix, docs).
- The "results" folder should contain a file with final model weights and a file describing the work that has been done.
- Include a README file in your project directory and a SUMMARY file in the results subdirectory.

