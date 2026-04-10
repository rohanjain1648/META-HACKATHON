# ForgeAI — Configuration

1. **Step 1: Install Dependencies**
   ```bash
   pip install -r forgeai/requirements.txt
   ```

2. **Step 2: Set up Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

3. **Step 3: Run the Framework**
   ```bash
   python -m forgeai.main --spec "Build a simple Task Management API with logic for due dates"
   ```

4. **Step 4: (Optional) Web Dashboard**
   ```bash
   python -m forgeai.main --web
   ```
