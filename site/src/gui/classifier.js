/** 
 * GUI module.
 * @module gui
 */
var dwv = dwv || {};
/**
 * Namespace for GUI functions.
 * @class gui
 * @namespace dwv
 * @static
 */
dwv.gui = dwv.gui || {};
dwv.gui.base = dwv.gui.base || {};

/**
 * Append the classify HTML to the page.
 * @method appendClassifyboxHtml
 * @static
 */
dwv.gui.base.appendClassifyboxHtml = function()
{
    // classify select
    var featurevalue1select  = dwv.html.createHtmlSelect("feature1Select", ['1', '2', '3', '4', '5']);
    var featurevalue2select  = dwv.html.createHtmlSelect("feature2Select", ['1', '2', '3', '4', '5']);
    var featurevalue3select  = dwv.html.createHtmlSelect("feature3Select", ['1', '2', '3', '4', '5']);
    var featurevalue4select  = dwv.html.createHtmlSelect("feature4Select", ['1', '2', '3', '4', '5']);
    var featurevalue5select  = dwv.html.createHtmlSelect("feature5Select", ['1', '2', '3', '4', '5']);
    var featurevalue6select  = dwv.html.createHtmlSelect("feature6Select", ['1', '2', '3', '4', '5']);
    var featurevalue7select  = dwv.html.createHtmlSelect("feature7Select", ['1', '2', '3', '4', '5']);
    var featurevalue8select  = dwv.html.createHtmlSelect("feature8Select", ['1', '2', '3', '4', '5']);
    var featurevalue9select  = dwv.html.createHtmlSelect("feature9Select", ['1', '2', '3', '4', '5']);
    var featurevalue10select = dwv.html.createHtmlSelect("feature10Select", ['1', '2', '3', '4', '5']);
    var featurevalue11select = dwv.html.createHtmlSelect("feature11Select", ['1', '2', '3', '4', '5']);
    var featurevalue12select = dwv.html.createHtmlSelect("feature12Select", ['1', '2', '3', '4', '5']);
    var classifierSelect     = dwv.html.createHtmlSelect("classifierSelect", ['LIDC', 'INBREAST']);
    var br1 = document.createElement("br");
    var br2 = document.createElement("br");
    var br3 = document.createElement("br");
    var br4 = document.createElement("br");
    var br5 = document.createElement("br");
    var br6 = document.createElement("br");
    var br7 = document.createElement("br");
    var br8 = document.createElement("br");
    var br9 = document.createElement("br");
    var feature1label   = document.createTextNode("Subtlety");
    var feature2label   = document.createTextNode("Internal Structure");
    var feature3label   = document.createTextNode("Calcification");
    var feature4label   = document.createTextNode("Sphericity");
    var feature5label   = document.createTextNode("Margin");
    var feature6label   = document.createTextNode("Lobulation");
    var feature7label   = document.createTextNode("Spiculation");
    var feature8label   = document.createTextNode("Texture");
    var classifierLabel = document.createTextNode("Classifier");
    
    var classifyButton       = document.createElement("button");
    classifyButton.innerHTML = 'Classify';
    classifyButton.onclick   = dwv.gui.onClassifyClick;
    classifyButton.id        = 'classifybutton';
    
    // node
    var node = document.getElementById("classifylist");
    // clear it
    while(node.hasChildNodes()) {
        node.removeChild(node.firstChild);
    }
    
    // append
    node.appendChild(feature1label);
    node.appendChild(featurevalue1select);
    //node.appendChild(br1);
    
    node.appendChild(feature2label);
    node.appendChild(featurevalue2select);
    //node.appendChild(br2);
    
    node.appendChild(feature3label);
    node.appendChild(featurevalue3select);
    //node.appendChild(br3);
    
    node.appendChild(feature4label);
    node.appendChild(featurevalue4select);
    //node.appendChild(br4);
    
    node.appendChild(feature5label);
    node.appendChild(featurevalue5select);
    //node.appendChild(br5);
    
    node.appendChild(feature6label);
    node.appendChild(featurevalue6select);
    //node.appendChild(br6);
    
    node.appendChild(feature7label);
    node.appendChild(featurevalue7select);
    //node.appendChild(br7);
    
    node.appendChild(feature8label);
    node.appendChild(featurevalue8select);
    //node.appendChild(br8);
    
    node.appendChild(classifierLabel);
    node.appendChild(classifierSelect);
    //node.appendChild(br9);
    
    node.appendChild(classifyButton);
    
    
    // trigger create event (mobile)
    $("#classifylist").trigger("create");
};

/**
 * Display the classifier HTML.
 * @method displayClassifyHtml
 * @static
 * @param {Boolean} bool True to display, false to hide.
 */
dwv.gui.base.displayClassifyHtml = function(bool)
{
    // file div element
    var filediv = document.getElementById("imagefilesdiv");
    filediv.style.display = bool ? "" : "none";
};