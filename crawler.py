import base64
import re
from typing import List, Tuple
import requests
from custom_dataset import CustomDataset


class GitHubTodoCrawler:
    def __init__(self, token: str, dataset: CustomDataset):
        self.github_api_url = "https://api.github.com"
        self.headers = {"Authorization": f"token {token}"}
        self.dataset = dataset

    def search_repos(self, query: str) -> List[str]:
        """Search repositories by keyword and return list of repository names."""
        repos = []
        url = f"{self.github_api_url}/search/repositories"
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 5}
        response = requests.get(url, headers=self.headers, params=params)
        response_data = response.json()
        for repo in response_data.get("items", []):
            repos.append(repo["full_name"])
        return repos

    def get_python_files(self, repo: str) -> List[str]:
        """Get all Python file paths in a given repository."""
        files = []
        url = f"{self.github_api_url}/repos/{repo}/git/trees/main?recursive=1"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            tree = response.json().get("tree", [])
            for file in tree:
                if file["path"].endswith(".py"):
                    files.append(file["path"])
        return files

    def get_file_content(self, repo: str, file_path: str) -> str:
        """Retrieve the content of a file from a GitHub repository."""
        url = f"{self.github_api_url}/repos/{repo}/contents/{file_path}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            content = response.json().get("content", "")
            # Decode the content from base64 and return it as a string
            return base64.b64decode(content).decode('utf-8')
        return ""

    def find_multiline_todo_comments(self, content: str) -> List[Tuple[int, str]]:
        """Find TODO comments in a Python file content, supporting multi-line TODO comments."""
        todos = []
        collecting_todo = False
        todo_text = ""
        start_line = 0

        for i, line in enumerate(content.splitlines(), start=1):
            if "TODO" in line.upper() and not collecting_todo:
                collecting_todo = True
                start_line = i
                todo_text = line.strip()
            elif collecting_todo:
                if line.lstrip().startswith("#"):
                    todo_text += self.extract_todo_text(line)
                else:
                    todos.append((start_line, todo_text))
                    collecting_todo = False
                    todo_text = ""

        if collecting_todo and todo_text:
            todos.append((start_line, todo_text))

        return todos

    def extract_todo_text(self, line: str) -> str:
        """Extracts the text following the 'TO DO:' marker from a comment line."""
        # Match 'TO DO' with optional characters in between, followed by ':' and capture everything after it
        match = re.search(r".*TODO[^:]*:\s*(.*)", line, re.IGNORECASE)
        if match:
            return match.group(1).strip()  # Return the captured text after TO DO and :
        return ""

    def process_repos(self, query: str):
        todos_to_annotate = []
        file_count = 0
        repos = self.search_repos(query)
        for repo in repos:
            python_files = self.get_python_files(repo)
            print(f"\nRepository {repo} :{len(python_files)} files to process")
            for file in python_files:
                content = self.get_file_content(repo, file)
                todos = self.find_multiline_todo_comments(content)
                todos_to_annotate.extend(todos)
                file_count += 1
                if file_count % 500 == 0 and todos_to_annotate:
                    seen_todos = set()
                    todos_to_annotate = [
                        todo for todo in todos_to_annotate
                        if todo[1].strip() and not self.dataset.entry_exists(todo[1]) and (
                                    todo[1] not in seen_todos and not seen_todos.add(todo[1]))
                    ]
                    print(f"\nNo of todos to annotate after processing {file_count} files: {len(todos_to_annotate)}")
                    for line_number, todo in todos_to_annotate:
                        priority = self.ask_for_annotation(line_number, todo)
                        self.dataset.add_entry(todo, priority)
                    todos_to_annotate = []

    def ask_for_annotation(self, line_number: int, todo: str) -> int:
        while True:
            try:
                priority = int(input(f"Enter priority for TODO: '{todo}' (1 to 4): "))
                if 1 <= priority <= 4:
                    break
                else:
                    print("Please enter a number between 1 and 4.")
            except ValueError:
                print("Invalid input. Please enter a valid integer between 1 and 4.")
        print(f"Line {line_number}: Priority {priority} - {todo}")
        return priority

COMET_API_KEY = "<COMET_API_KEY>"
COMET_PROJECT_NAME = "<COMET_PROJECT_NAME>"
COMET_WORKSPACE = "915-muscalagiu-ancaioana"
dataset = CustomDataset(COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE)
GITHUB_TOKEN = "<GITHUB_TOKEN>"
crawler = GitHubTodoCrawler(GITHUB_TOKEN, dataset)
crawler.process_repos("machine learning")