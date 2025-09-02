# WordPress Author Selection Guide

This document provides instructions on how to use different WordPress authors when posting articles using the CopyscriptAI system.

## Default Author Configuration

By default, the system uses the author ID specified in the `.env` file with the `WP_AUTHOR` variable. This is usually set to `1`, which corresponds to the admin user in most WordPress installations.

## Viewing Available WordPress Authors

To see a list of available WordPress users that can be used as authors, run the `list_wordpress_users.py` script:

```bash
python list_wordpress_users.py
```

This will display a table with the following information for each user:
- ID: The numeric ID of the user (this is what you'll use as the author ID)
- Username: The user's login username
- Name: The user's display name
- Role: The user's role(s) in WordPress (admin, editor, author, etc.)

## Using a Custom Author

There are two ways to specify a custom author:

### Method 1: Set in .env File (Permanent)

Add or modify the `WP_CUSTOM_AUTHOR` variable in your `.env` file:

```
WP_CUSTOM_AUTHOR=2
```

This will override the default author (specified by `WP_AUTHOR`) for all article generations until you remove or comment out this line.

### Method 2: Environment Variable (Temporary)

You can also set the custom author just for a single run by using an environment variable when executing the script:

```bash
WP_CUSTOM_AUTHOR=2 python main.py input.csv
```

This will use author ID 2 for that specific run without changing your `.env` file.

## How It Works

When generating articles:

1. The system first checks for a `WP_CUSTOM_AUTHOR` setting (either in the environment or `.env` file)
2. If found, it uses that author ID instead of the default `WP_AUTHOR` value
3. The system will display a message indicating which author ID is being used

## Troubleshooting

If you encounter issues with author selection:

1. Make sure the author ID exists in your WordPress installation
2. Confirm the user has appropriate permissions to create posts
3. Check that your WordPress API credentials have permission to post as other users

For additional help, run `list_wordpress_users.py` to verify the available user IDs.
