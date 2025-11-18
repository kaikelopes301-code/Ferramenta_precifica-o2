name: Performance Monitoring & Gates

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  python-performance:
    name: Python Performance Tests
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install Python dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt

      # Se quiser rodar testes Python, coloque aqui:
      #- name: Run Python tests
      #  run: pytest

  
  frontend-performance:
    name: Frontend Performance Tests
    runs-on: ubuntu-latest
    timeout-minutes: 20
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: 'frontend/package-lock.json'
      
      - name: Install dependencies
        working-directory: ./frontend
        run: npm ci
      
      - name: Build & Analyze Bundle
        working-directory: ./frontend
        env:
          ANALYZE: 'true'
        run: |
          npm run build
          
          echo "📊 Bundle Analysis Results:"
          
          if ls .next/static/chunks/pages/_app-*.js 1> /dev/null 2>&1; then
            APP_SIZE=$(du -sb .next/static/chunks/pages/_app-*.js | awk '{print $1}')
            echo "App bundle size: ${APP_SIZE} bytes"
            
            MAX_APP_SIZE=328000
            if [ "$APP_SIZE" -gt "$MAX_APP_SIZE" ]; then
              echo "❌ App bundle too large: ${APP_SIZE}B > ${MAX_APP_SIZE}B"
              exit 1
            fi
          fi
          
          TOTAL_JS_SIZE=$(find .next/static -name "*.js" -type f -exec du -sb {} + | awk '{sum+=$1} END {print sum}')
          echo "Total JS size: ${TOTAL_JS_SIZE} bytes"
          
          MAX_TOTAL_SIZE=2000000 
          if [ "$TOTAL_JS_SIZE" -gt "$MAX_TOTAL_SIZE" ]; then
            echo "❌ Total JS too large: ${TOTAL_JS_SIZE}B > ${MAX_TOTAL_SIZE}B"
            exit 1
          fi
          
          echo "✅ Bundle size checks passed"
