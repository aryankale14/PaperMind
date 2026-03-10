const { Client } = require('pg');
require('dotenv').config();

const client = new Client({
    connectionString: process.env.DATABASE_URL,
});

async function run() {
    try {
        await client.connect();
        console.log("Connected to Supabase.");

        await client.query("DROP TABLE IF EXISTS document_chunks;");
        console.log("Dropped table document_chunks.");

    } catch (err) {
        console.error("Error executing query", err.stack);
    } finally {
        await client.end();
    }
}

run();
