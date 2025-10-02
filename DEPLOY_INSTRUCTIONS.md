# ShareConvo Cloudflare Deployment Instructions

This guide will walk you through deploying the ShareConvo application to the Cloudflare network using the Wrangler CLI.

## ‼️ Important Prerequisite: The Database ‼️

This application is built with Flask and Flask-SQLAlchemy, a library that requires a standard database connection. Cloudflare's D1 database uses a custom API that is not natively compatible with SQLAlchemy.

**The `_worker.py` entrypoint in this repository will NOT work out-of-the-box because of this database incompatibility.**

To make this application work on Cloudflare, you would need to:
1.  **Find or create a SQLAlchemy dialect for Cloudflare D1.** This is a piece of software that translates between SQLAlchemy's commands and D1's API. You may find community-driven projects for this.
2.  **Modify the `_worker.py` script** to use this dialect to patch the database connection before the Flask app is run.

The following instructions assume that the database issue is solved and show you how to proceed with the deployment process itself.

## Step 1: Install Prerequisites

You will need Node.js and `npm` installed on your machine. You can get them from [https://nodejs.org/](https://nodejs.org/).

## Step 2: Install Cloudflare Wrangler

Wrangler is the command-line tool for managing Cloudflare Workers and Pages projects. Install it globally by running:

```bash
npm install -g wrangler
```

## Step 3: Authenticate with Cloudflare

Log in to your Cloudflare account by running:

```bash
wrangler login
```

This will open a browser window for you to authorize Wrangler.

## Step 4: Create the D1 Database

You need to create a D1 database in the Cloudflare dashboard.

1.  Go to your Cloudflare Dashboard.
2.  Navigate to **Workers & Pages** > **D1**.
3.  Click **Create database**.
4.  Enter a database name, for example `shareconvo-db`.
5.  Select a region.
6.  After creation, you will get a **Database ID**.

## Step 5: Configure `wrangler.toml`

Open the `wrangler.toml` file in this repository. You need to fill in the `database_id` you received from the previous step.

```toml
[[d1_databases]]
binding = "DB"
database_name = "shareconvo-db"
database_id = "xxxx-xxxx-xxxx-xxxx-xxxx" # <--- PASTE YOUR ID HERE
preview_database_id = "xxxx-xxxx-xxxx-xxxx-xxxx" # <--- PASTE YOUR PREVIEW DB ID HERE
```

You will also need your Cloudflare **Account ID**. You can find this on the main overview page of your Cloudflare dashboard. You may need to add it to the top of the `wrangler.toml` file:
```toml
account_id = "YOUR_ACCOUNT_ID"
```

## Step 6: Set the Application Secret

The Flask application needs a `SECRET_KEY` for signing sessions. You must set this as a secret in your Cloudflare project.

Run the following command, replacing `YOUR_SUPER_SECRET_KEY` with a long, random string:

```bash
wrangler secret put SECRET_KEY
```

Wrangler will prompt you to enter the secret value.

## Step 7: Deploy the Application

Once everything is configured, you can deploy the application with a single command:

```bash
wrangler deploy
```

If the deployment is successful, Wrangler will output the URL where your application is live (e.g., `https://shareconvo.your-worker-subdomain.workers.dev`).
