import praw
import datetime
import pandas as pd
import os


reddit = praw.Reddit(
    client_id="your_client_id",
    client_secret="your_client_secret",
    user_agent="your_user_agent"
)


subreddits = ["adhd", "anxiety", "bipolar", "depression", "ptsd"]

all_posts = []

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)

    for submission in subreddit.new(limit=None):
        title = submission.title
        content = submission.selftext
        author = submission.author.name if submission.author else "[deleted]"
        created_time = datetime.datetime.fromtimestamp(submission.created_utc)

        all_posts.append({
            "Author": author,
            "Posted at": created_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Title": title,
            "Content": content,
            "Subreddit(Label of category)": subreddit_name
        })


output_folder = "data"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "multi_subreddit_posts.xlsx")

df = pd.DataFrame(all_posts)
df.to_excel(output_file, index=False)

print(os.path.abspath(output_file))
