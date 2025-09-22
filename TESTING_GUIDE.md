# 🧪 Ethical Eye Extension Testing Guide

## 🚀 **Step 1: Start the API Server**

Open a terminal and run:
```bash
python api/ethical_eye_api.py
```

You should see:
```
🚀 ETHICAL EYE API SERVER
============================================================
Research Project: Explainable AI for Dark Pattern Detection
Server started at: http://127.0.0.1:5000
============================================================
Loading trained DistilBERT model...
Model loaded successfully!
Number of labels: 8
Labels: ['Forced Action', 'Misdirection', 'Not Dark Pattern', 'Obstruction', 'Scarcity', 'Sneaking', 'Social Proof', 'Urgency']
SHAP explainer initialized successfully!
✅ API Server is running and ready!
```

## 🔧 **Step 2: Load the Chrome Extension**

### 2.1 Open Chrome Extensions Page
1. Open Chrome browser
2. Go to `chrome://extensions/`
3. Enable **Developer mode** (toggle in top-right)

### 2.2 Load Your Extension
1. Click **"Load unpacked"**
2. Navigate to your project folder
3. Select the `app` folder (contains manifest.json)
4. Click **"Select Folder"**

### 2.3 Verify Extension is Loaded
- You should see "Ethical Eye" in your extensions list
- The extension icon should appear in your Chrome toolbar

## 🌐 **Step 3: Test on Real Websites**

### 3.1 Test Websites with Dark Patterns

**E-commerce Sites (High Dark Pattern Content):**
- **Amazon**: Look for urgency ("Only 2 left!"), scarcity ("Limited time!")
- **Booking.com**: Check for social proof ("12 people viewing")
- **Airbnb**: Look for urgency and scarcity patterns
- **Shopify stores**: Various dark patterns

**News/Social Sites:**
- **BuzzFeed**: Clickbait headlines
- **Facebook**: Social proof notifications
- **LinkedIn**: Urgency in job postings

### 3.2 How to Use the Extension

1. **Navigate** to any website
2. **Click** the Ethical Eye extension icon in your toolbar
3. **Click "Analyze Page"** button
4. **Wait** for analysis (should take 2-5 seconds)
5. **View results** in the popup

### 3.3 What to Look For

**In the Popup:**
- **Total Patterns Found**: Number of dark patterns detected
- **Categories**: List of pattern types found
- **Confidence Scores**: How confident the model is
- **Explanations**: SHAP-based explanations

**On the Webpage:**
- **Highlighted Text**: Dark patterns should be highlighted
- **Tooltips**: Hover over highlights to see details
- **Color Coding**: Different colors for different pattern types

## 📊 **Step 4: Test Different Pattern Types**

### 4.1 Urgency Patterns
**Test Text**: "Hurry! Only 2 left in stock!"
**Expected**: Should detect as "Urgency" with high confidence

### 4.2 Scarcity Patterns  
**Test Text**: "Limited time offer - expires soon!"
**Expected**: Should detect as "Scarcity" with high confidence

### 4.3 Social Proof Patterns
**Test Text**: "Join 10,000+ satisfied customers"
**Expected**: Should detect as "Social Proof" with high confidence

### 4.4 Misdirection Patterns
**Test Text**: "Click here for free shipping" (but leads to paid options)
**Expected**: Should detect as "Misdirection"

### 4.5 Normal Content
**Test Text**: "Welcome to our website"
**Expected**: Should detect as "Not Dark Pattern"

## 🔍 **Step 5: Detailed Testing Checklist**

### ✅ **Functionality Tests**
- [ ] Extension loads without errors
- [ ] API server responds to requests
- [ ] Popup opens when clicking extension icon
- [ ] "Analyze Page" button works
- [ ] Analysis completes within 10 seconds
- [ ] Results display in popup
- [ ] Webpage text gets highlighted
- [ ] Tooltips show on hover

### ✅ **Accuracy Tests**
- [ ] Urgency patterns detected correctly
- [ ] Scarcity patterns detected correctly
- [ ] Social proof patterns detected correctly
- [ ] Normal content not flagged as dark patterns
- [ ] Confidence scores are reasonable (0.1-1.0)
- [ ] Explanations make sense

### ✅ **Performance Tests**
- [ ] Analysis completes quickly (< 10 seconds)
- [ ] No browser freezing or crashes
- [ ] Memory usage is reasonable
- [ ] API responds consistently

## 🐛 **Step 6: Troubleshooting**

### Common Issues:

**1. Extension Not Loading**
- Check if manifest.json is valid
- Ensure all files are in the `app` folder
- Try refreshing the extensions page

**2. API Connection Failed**
- Verify API server is running on port 5000
- Check if `http://127.0.0.1:5000` is accessible
- Look for CORS errors in browser console

**3. No Patterns Detected**
- Check if the webpage has text content
- Verify the model is loaded correctly
- Check browser console for errors

**4. Slow Performance**
- Check API server logs
- Verify GPU is being used (if available)
- Check network connectivity

## 📈 **Step 7: Performance Monitoring**

### Monitor These Metrics:
- **Response Time**: How long analysis takes
- **Accuracy**: Are patterns detected correctly?
- **False Positives**: Normal content flagged incorrectly
- **False Negatives**: Dark patterns missed
- **User Experience**: Is the interface intuitive?

## 🎯 **Step 8: User Study Preparation**

### For Your Research Paper:
1. **Record Results**: Take screenshots of detections
2. **Note Accuracy**: Track correct vs incorrect detections
3. **User Feedback**: Ask testers about the interface
4. **Performance Data**: Record response times
5. **Edge Cases**: Test unusual websites

## 🚀 **Step 9: Advanced Testing**

### Test Edge Cases:
- **Very Long Pages**: Test with content-heavy sites
- **Dynamic Content**: Test with JavaScript-heavy sites
- **Different Languages**: Test with non-English content
- **Mobile Sites**: Test responsive design
- **Different Browsers**: Test in Chrome, Edge, Firefox

## 📝 **Step 10: Documentation**

### Record Your Findings:
- **Test Results**: Which patterns were detected correctly
- **Performance Metrics**: Response times and accuracy
- **User Feedback**: What testers think about the tool
- **Issues Found**: Any bugs or problems
- **Improvements**: Suggestions for enhancement

---

## 🎉 **Ready to Test!**

Your Ethical Eye extension is now ready for comprehensive testing. The trained model achieved **88.15% accuracy**, so you should see excellent results on real websites!

**Start with the API server, then load the extension and test on e-commerce sites with known dark patterns.**

Good luck with your testing! 🚀
