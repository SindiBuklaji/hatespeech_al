import instaloader
import csv
L = instaloader.Instaloader()

L.login('sindibuklaj', '26072002Lezha!')

profile = instaloader.Profile.from_username(L.context, 'joqnews')

counter = 0

with open('comments_dataset.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Define the header of your CSV file
    writer.writerow(['Post Shortcode', 'Commenter Username', 'Comment Text'])


    counter = 0
    for post in profile.get_posts():
        if counter < 10:  # Limit to first 10 posts
            print(f"Accessing comments from post {post.shortcode}")
            for comment in post.get_comments():
                # Write each comment to the CSV file
                writer.writerow([post.shortcode, comment.owner.username, comment.text])
            counter += 1
        else:
            break
