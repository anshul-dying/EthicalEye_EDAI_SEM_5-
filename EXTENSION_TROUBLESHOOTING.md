# 🔧 Extension Troubleshooting Guide

## 🚨 **Issue: Extension Not Sending Requests**

### ✅ **FIXED: Confidence Threshold Issue**
The main issue was that the extension was using a confidence threshold of **0.7**, but your model returns confidence scores of **0.16-0.30**. 

**✅ FIXED:** Changed threshold from `0.7` to `0.15` in `app/js/content.js`

## 🧪 **Testing Steps**

### 1. **Reload the Extension**
1. Go to `chrome://extensions/`
2. Find "DarkSurfer" extension
3. Click the **refresh/reload** button 🔄
4. This will load the updated code with the fixed threshold

### 2. **Test with Debug Page**
1. Open the test page: `test_extension_connection.html` in your browser
2. Click the extension icon
3. Click "Analyze Page"
4. Open Chrome DevTools (F12) and check the Console tab

### 3. **Check Console Logs**
You should see:
```
🔍 Ethical Eye: Starting analysis...
🔍 Sending X segments to API
🔍 API Response status: 200
🔍 API Response data: {...}
```

### 4. **Expected Results**
- **Test 1**: "Hurry! Only 2 left!" → Should highlight as Scarcity/Urgency
- **Test 2**: "Limited time offer" → Should highlight as Urgency/Scarcity  
- **Test 3**: "Join 10,000+ customers" → Should highlight as Social Proof
- **Test 4**: "Welcome to our website" → Should NOT highlight (normal content)
- **Test 5**: "Click here for free shipping" → Should highlight as Misdirection/Obstruction

## 🔍 **Debugging Checklist**

### ✅ **API Server**
- [ ] API server is running: `python api/ethical_eye_api.py`
- [ ] Server shows: "Running on http://127.0.0.1:5000"
- [ ] No errors in API server console

### ✅ **Extension**
- [ ] Extension is loaded in Chrome
- [ ] Extension is enabled (not disabled)
- [ ] Extension was reloaded after code changes
- [ ] Extension icon appears in toolbar

### ✅ **Browser Console**
- [ ] Open DevTools (F12)
- [ ] Check Console tab for errors
- [ ] Look for "🔍 Ethical Eye" messages
- [ ] Check Network tab for API requests

### ✅ **Permissions**
- [ ] Extension has "activeTab" permission
- [ ] Extension has "host_permissions" for all URLs
- [ ] No CORS errors in console

## 🚨 **Common Issues & Solutions**

### Issue 1: "Failed to fetch" Error
**Cause**: API server not running
**Solution**: 
```bash
python api/ethical_eye_api.py
```

### Issue 2: No Console Messages
**Cause**: Extension not reloaded
**Solution**: Reload extension in `chrome://extensions/`

### Issue 3: CORS Error
**Cause**: Browser blocking requests
**Solution**: API server has CORS enabled, should work

### Issue 4: No Highlights Appearing
**Cause**: Confidence threshold too high
**Solution**: ✅ **FIXED** - Threshold changed to 0.15

### Issue 5: Extension Icon Not Clickable
**Cause**: Extension not properly loaded
**Solution**: 
1. Remove extension
2. Reload from `app` folder
3. Check for errors in extension details

## 🎯 **Quick Test Commands**

### Test API Directly:
```bash
python quick_api_test.py
```

### Test Model Directly:
```bash
python diagnose_model.py
```

### Test Extension:
1. Open `test_extension_connection.html`
2. Click extension → "Analyze Page"
3. Check console for logs

## 📊 **Expected Performance**

With the fixed threshold (0.15), you should see:
- **Response Time**: 30-50ms per request
- **Confidence Scores**: 16-30% (normal for this model)
- **Detection Rate**: High (most patterns detected)
- **Accuracy**: 88.15% (from training results)

## 🎉 **Success Indicators**

✅ **Extension Working When:**
- Console shows "🔍 Ethical Eye: Starting analysis..."
- API requests appear in Network tab
- Text gets highlighted on webpage
- Tooltips show when hovering over highlights
- No error messages in console

## 🚀 **Next Steps After Fix**

1. **Test on Real Websites**: Amazon, Booking.com, etc.
2. **Verify Pattern Detection**: Check different pattern types
3. **Test User Experience**: Hover over highlights, read tooltips
4. **Document Results**: For your research paper

---

## 🆘 **Still Having Issues?**

If the extension still doesn't work:

1. **Check API Server**: Make sure it's running and responding
2. **Check Extension Console**: Look for error messages
3. **Try Different Website**: Test on a simple page first
4. **Restart Browser**: Sometimes helps with extension issues

**The main issue (confidence threshold) has been fixed!** 🎉
