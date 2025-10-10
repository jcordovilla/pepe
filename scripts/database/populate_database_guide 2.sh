#!/bin/bash
"""
Complete Discord Data Population Guide

This script guides you through populating your database with actual Discord message data.
Run this after setting up your Discord bot token and permissions.
"""

echo "🤖 Discord Bot Database Population Guide"
echo "=========================================="
echo ""

echo "📋 Prerequisites:"
echo "1. ✅ Discord bot token in .env file (DISCORD_TOKEN)"
echo "2. ✅ Bot has Read Message History permission"
echo "3. ✅ Bot is added to your Discord server"
echo "4. ✅ OpenAI API key in .env file (OPENAI_API_KEY)"
echo ""

echo "🚀 Step 1: Fetch Messages from Discord"
echo "--------------------------------------"
echo "This will collect all messages from your Discord server:"
echo ""
echo "python core/fetch_messages.py"
echo ""
echo "📁 Output: Messages saved to data/fetched_messages/*.json"
echo "⏱️  Time: ~2-5 minutes for a typical server"
echo "💾 Storage: ~1-10MB depending on message volume"
echo ""

echo "🔄 Step 2: Process and Store in Vector Database"
echo "----------------------------------------------"
echo "This will embed and index all fetched messages:"
echo ""
echo "python core/embed_store.py"
echo ""
echo "📁 Output: Messages stored in ChromaDB (data/chromadb/)"
echo "⏱️  Time: ~5-15 minutes depending on message count"
echo "💾 Storage: ~10-100MB for embeddings"
echo ""

echo "✅ Step 3: Verify Database Population"
echo "------------------------------------"
echo "Test that your database has real data:"
echo ""
echo "python test_database_search.py"
echo ""

echo "🎉 Step 4: Test Complete Bot Integration"
echo "---------------------------------------"
echo "Verify the bot works with real data:"
echo ""
echo "python test_discord_bot_complete.py"
echo ""

echo "📊 Optional: Check Database Stats"
echo "--------------------------------"
echo "python check_vector_store.py"
echo ""

echo "🚀 Start the Bot for Production"
echo "------------------------------"
echo "python main.py"
echo ""

echo "✨ Pro Tips:"
echo "- The bot supports incremental message fetching (only new messages)"
echo "- Reaction data is captured and indexed automatically"
echo "- The system scales to handle thousands of messages"
echo "- All message metadata (authors, timestamps, jump URLs) is preserved"
echo ""

echo "🔧 If you encounter issues:"
echo "- Check Discord bot permissions"
echo "- Verify .env file has all required tokens"
echo "- Check logs in logs/ directory"
echo "- Run individual test scripts to isolate issues"
