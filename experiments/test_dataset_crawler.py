import os
from git import Repo
from difflib import SequenceMatcher

class TodoCrawler:
    def __init__(self, repo_url, local_dir, previous_tag, current_tag):
        self.repo_url = repo_url
        self.local_dir = local_dir
        self.previous_tag = previous_tag
        self.current_tag = current_tag

    def clone_repo(self):
        """Clone the repository if it doesn't already exist."""
        if not os.path.exists(self.local_dir):
            print(f"Cloning repository from {self.repo_url}...")
            Repo.clone_from(self.repo_url, self.local_dir)
        else:
            print("Repository already cloned.")

    def checkout_and_extract_todos(self, tag):
        """Checkout a specific tag and extract TODOs."""
        repo = Repo(self.local_dir)
        repo.git.checkout(tag)
        todos = self.extract_todos(self.local_dir)
        return todos

    def extract_todos(self, directory):
        """Extract TODO comments from Python files in a directory."""
        todos = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if 'TODO' in line:
                                todos.append({
                                    'file': os.path.relpath(filepath, directory),
                                    'line': i + 1,
                                    'text': line.strip()
                                })
        return todos

    def compare_todos(self, todos_prev, todos_curr, threshold=0.2):
        """Compare TODOs between two releases to find high-priority ones."""
        low_priority_todos = []
        for todo_prev in todos_prev:
            for todo_curr in todos_curr:
                if todo_prev['file'] == todo_curr['file']:
                    if self.is_similar(todo_prev['text'], todo_curr['text'], threshold):
                        low_priority_todos.append(todo_prev)
                        break
        return low_priority_todos

    @staticmethod
    def is_similar(text1, text2, threshold):
        """Check if two to do texts are similar above a given threshold."""
        return SequenceMatcher(None, text1, text2).ratio() > threshold

    def run(self):
        # Clone the repo if not already cloned
        self.clone_repo()

        # Checkout and extract TODOs from previous release
        print(f"Checking out and extracting TODOs from {self.previous_tag}...")
        todos_prev = self.checkout_and_extract_todos(self.previous_tag)

        # Checkout and extract TODOs from current release
        print(f"Checking out and extracting TODOs from {self.current_tag}...")
        todos_curr = self.checkout_and_extract_todos(self.current_tag)

        # Compare TODOs to find high-priority ones
        print("Comparing TODOs between releases...")
        low_priority_todos = self.compare_todos(todos_prev, todos_curr)

        # Step 5: Create dataset
        dataset = [{'todo_text': todo['text'], 'priority': 1} for todo in low_priority_todos] + \
                  [{'todo_text': todo['text'], 'priority': 0} for todo in todos_prev if todo not in low_priority_todos]

        return dataset



if __name__ == "__main__":
    # Define parameters
    repo_url = "https://github.com/keras-team/keras.git"  # keras repository URL
    local_dir = "./keras_repo"
    previous_tag = "v2.15.0"
    current_tag = "v3.0.0"

    # Initialize and run the crawler
    crawler = TodoCrawler(repo_url, local_dir, previous_tag, current_tag)
    dataset = crawler.run()

    # Save dataset to CSV
    import pandas as pd
    df = pd.DataFrame(dataset)
    df.to_csv("test_todo_dataset.csv", index=False)

    print("Dataset saved to test_todo_dataset.csv")
