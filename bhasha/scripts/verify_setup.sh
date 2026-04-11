#!/bin/bash
# Quick verification script to check if everything is set up correctly

echo "=========================================="
echo "BhashaLLM Setup Verification"
echo "=========================================="
echo ""

# Check if models directory exists
if [ -d "models/base_models" ]; then
    echo "✅ Base models directory exists"
    echo "   Size: $(du -sh models/base_models | cut -f1)"
else
    echo "❌ Base models directory NOT found"
    echo "   Run: python3 download_base_models.py"
fi

# Check if adapters exist
if [ -d "models/bangla_adapters/final_adapter" ]; then
    echo "✅ Bangla LLM adapter exists"
else
    echo "❌ Bangla LLM adapter NOT found"
fi

if [ -d "models/instruct_adapters/final_instruct_adapter" ]; then
    echo "✅ Instruction adapter exists"
else
    echo "❌ Instruction adapter NOT found"
fi

if [ -d "models/ocr_adapters/banglawriting_adapter" ]; then
    echo "✅ OCR adapter exists"
else
    echo "❌ OCR adapter NOT found"
fi

echo ""

# Check if virtual environment is activated
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Virtual environment is activated"
    echo "   Path: $VIRTUAL_ENV"
else
    echo "⚠️  Virtual environment NOT activated"
    echo "   Run: source venv/bin/activate"
fi

echo ""

# Check if models are in gitignore
if git check-ignore models/ > /dev/null 2>&1; then
    echo "✅ models/ is properly ignored by Git"
else
    echo "⚠️  models/ might not be ignored by Git"
fi

echo ""
echo "=========================================="
echo "Total Storage Used:"
echo "=========================================="
du -sh models/* 2>/dev/null | sort -h

echo ""
echo "=========================================="
echo "Quick Test:"
echo "=========================================="
echo "Run: python3 test_models.py"
echo "     python3 model_paths.py"
echo ""
