# Projects

## Project One Name
Brand Veda 
An application for a SaaS platform that tracks brand visibility across AI assistants — ChatGPT (GPT-4o), Perplexity AI, and Google Gemini. MVP is focused on skincare brands.

How It Works
User completes an 8-step onboarding (brand info, category, competitors, target audience)
System auto-generates 18 prompts across awareness, consideration, and decision stages
User confirms → background job fires 54 LLM calls (18 prompts × 3 providers) in parallel
Responses are parsed for brand mentions, position, and sentiment
Four scores are calculated per provider and averaged into a combined score
Dashboard shows visibility trends, share of voice, and competitor comparisons
Weekly CRON re-runs analysis and emails a digest to active subscribers

## Project Two Name

CareerBrain – AI-powered career platform (MVP) | NestJS, TypeScript, OpenRouter, MongoDB NextJS
• Developing an AI career copilot that models profiles, matches candidates to opportunities, and generates personalized application assets.

## Ask Mahima (this project!)

A personal RAG chatbot built with FastAPI, ChromaDB, sentence-transformers embeddings, and an LLM via OpenRouter, that answers questions about Mahima using only her personal knowledge base.
