# WordPress Author Selection

This document explains how to select different WordPress authors when posting articles.

## Background

Previously, the script only allowed posting with the default WordPress author specified in the `.env` file. Now, you can select different authors for your posts.

## Listing Available WordPress Authors

To see a list of available WordPress authors, run:

```bash
python list_wordpress_users.py
```

This will display a table with all WordPress users, including their IDs, names, and roles.

## Setting a Custom WordPress Author

There are two ways to specify a custom WordPress author:

### Method 1: Environment Variable (for all articles in a batch)

To use the same custom author for all articles in a batch, set the `WP_CUSTOM_AUTHOR` environment variable before running the script:

```bash
# Set the custom author ID (replace '2' with your desired author ID)
export WP_CUSTOM_AUTHOR=2

# Run the script with this author
python main.py keywords.csv
```

### Method 2: Default Author in .env file

If you don't specify a custom author, the script will use the default author specified in your `.env` file:

```
WP_AUTHOR=1
```

## Finding WordPress User IDs

1. Run `python list_wordpress_users.py` to see all available WordPress users and their IDs
2. Alternatively, check your WordPress admin panel:
   - Go to Users → All Users
   - Hover over a user's name and look at the URL in your browser's status bar
   - The ID will be visible in the URL (e.g., `user-edit.php?user_id=2`)

## Troubleshooting

If you get permission errors when posting, ensure that:
1. The WordPress user has appropriate publishing permissions
2. Your application password has the correct capabilities
3. The user exists in your WordPress installation
