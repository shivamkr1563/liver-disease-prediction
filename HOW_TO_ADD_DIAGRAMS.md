# How to Add Diagrams to Your IEEE Conference Paper

## ✅ Step 1: Generate the Figures (COMPLETED)

All 6 required figures have been generated in your project folder:
- ✓ architecture_diagram.png
- ✓ confusion_matrices.png
- ✓ roc_curves.png
- ✓ feature_importance.png
- ✓ kan_splines.png
- ✓ webapp_interface.png

## 📝 Step 2: Compile Your LaTeX Paper

### Option A: Using Overleaf (Recommended for beginners)
1. Go to https://www.overleaf.com/
2. Create a free account or log in
3. Click "New Project" → "Upload Project"
4. Upload your `IEEE_Conference_Paper.tex` file
5. Upload all 6 PNG image files to the same project
6. Click "Recompile" button
7. Download the PDF

### Option B: Using Local LaTeX Installation

**Windows (MiKTeX):**
```powershell
# Install MiKTeX from: https://miktex.org/download
# After installation, compile:
cd "c:\Users\shiva\Desktop\Liver Disease Prediction(KAN)"
pdflatex IEEE_Conference_Paper.tex
bibtex IEEE_Conference_Paper
pdflatex IEEE_Conference_Paper.tex
pdflatex IEEE_Conference_Paper.tex
```

**Using VS Code with LaTeX Workshop Extension:**
1. Install "LaTeX Workshop" extension in VS Code
2. Open `IEEE_Conference_Paper.tex`
3. Press `Ctrl+Alt+B` to build
4. View PDF with `Ctrl+Alt+V`

## 🔍 Step 3: Verify All Figures Appear

After compilation, check that all 6 figures appear correctly:
- Figure 1: System Architecture (page ~4)
- Figure 2: Confusion Matrices (page ~7)
- Figure 3: ROC Curves (page ~7)
- Figure 4: Feature Importance (page ~8)
- Figure 5: KAN Splines (page ~8)
- Figure 6: Web Interface (page ~9)

## 🎨 Step 4: Customize Figures (Optional)

To modify any figure, edit `generate_figures.py`:
```python
# Example: Change figure size
plt.subplots(figsize=(12, 8))  # width, height in inches

# Example: Change colors
color='#FF5733'  # Use hex color codes

# Example: Adjust text
fontsize=14, fontweight='bold'
```

Then regenerate:
```powershell
python generate_figures.py
```

## 📊 Figure Details

### Figure 1: Architecture Diagram
- Shows complete system workflow
- Frontend → Backend → Preprocessing → ML Models → Results

### Figure 2: Confusion Matrices  
- Shows all 5 models (LR, RF, XGBoost, MLP, KAN)
- True/False Positives and Negatives
- Accuracy percentages

### Figure 3: ROC Curves
- Compares all 5 models
- Shows AUC scores
- Diagonal reference line

### Figure 4: Feature Importance
- Horizontal bar chart
- Top 10 clinical features
- Importance scores from Random Forest

### Figure 5: KAN Splines
- Three subplots (a, b, c)
- Shows learned transformations
- Bilirubin, Albumin, AST

### Figure 6: Web Application
- Mockup of user interface
- Input form with fields
- Results display section

## 🚨 Troubleshooting

**Problem: Images not found during compilation**
- Ensure all PNG files are in same folder as .tex file
- Check file names match exactly (case-sensitive on Linux/Mac)

**Problem: Low quality images in PDF**
- Images are created at 300 DPI (print quality)
- If blurry, regenerate with higher DPI in script

**Problem: Figures too large/small**
- Adjust `\columnwidth` in LaTeX
- Or modify `figsize` in Python script

**Problem: Can't compile LaTeX**
- Use Overleaf (easiest option)
- Or install MiKTeX + TeXstudio on Windows

## 📤 Submission Checklist

Before submitting to IEEE conference:
- [ ] All 6 figures appear correctly
- [ ] Figure captions are descriptive
- [ ] All figure references (Fig. 1, Fig. 2, etc.) work
- [ ] Images are high quality (300 DPI)
- [ ] Paper compiles without errors
- [ ] References section is complete (20 citations)
- [ ] Page limit met (typically 6-8 pages for IEEE)

## 🎓 Tips for IEEE Papers

1. **Figure Quality**: Always use 300 DPI for publication
2. **Color vs. Grayscale**: IEEE accepts color, but ensure figures work in B&W
3. **File Size**: Keep PDF under 10 MB for submission
4. **Caption Style**: Be descriptive, explain all elements
5. **In-text References**: Always refer to figures in the text

## 📧 Need Help?

If you encounter issues:
1. Check LaTeX log file for specific errors
2. Verify all packages are installed
3. Try compiling on Overleaf first (simplest)
4. Ensure all image files are present

---

**Your paper is ready for compilation!** 🎉

Just compile the LaTeX file and all figures will be automatically included.
