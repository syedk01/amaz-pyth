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

/**
 * Handle window/level change.
 * @method onChangeWindowLevelPreset
 * @namespace dwv.gui
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onChangeWindowLevelPreset = function(/*event*/)
{
    dwv.tool.updateWindowingDataFromName(this.value);
};

/**
 * Handle colour map change.
 * @method onChangeColourMap
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onChangeColourMap = function(/*event*/)
{
    dwv.tool.updateColourMapFromName(this.value);
};

/**
 * Handle loader change.
 * @method onChangeLoader
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onChangeLoader = function(/*event*/)
{
    if( this.value === "file") {
        dwv.gui.displayUrlLoadHtml(false);
        dwv.gui.displayFileLoadHtml(true);
    }
    else if( this.value === "url") {
        dwv.gui.displayFileLoadHtml(false);
        dwv.gui.displayUrlLoadHtml(true);
    }
};

/**
 * Handle classify button click.
 * @method onClassifyClick
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onClassifyClick = function(event)
{
	var featurevalue1select = document.getElementById("feature1Select");
	var featurevalue1       = featurevalue1select.options[featurevalue1select.selectedIndex].text;
	
	var featurevalue2select = document.getElementById("feature2Select");
	var featurevalue2       = featurevalue2select.options[featurevalue2select.selectedIndex].text;
	
	var featurevalue3select = document.getElementById("feature3Select");
	var featurevalue3       = featurevalue3select.options[featurevalue3select.selectedIndex].text;
	
	var featurevalue4select = document.getElementById("feature4Select");
	var featurevalue4       = featurevalue4select.options[featurevalue4select.selectedIndex].text;
	
	var featurevalue5select = document.getElementById("feature5Select");
	var featurevalue5       = featurevalue5select.options[featurevalue5select.selectedIndex].text;
	
	var featurevalue6select = document.getElementById("feature6Select");
	var featurevalue6       = featurevalue6select.options[featurevalue6select.selectedIndex].text;
	
	var featurevalue7select = document.getElementById("feature7Select");
	var featurevalue7       = featurevalue7select.options[featurevalue7select.selectedIndex].text;
	
	var featurevalue8select = document.getElementById("feature8Select");
	var featurevalue8       = featurevalue8select.options[featurevalue8select.selectedIndex].text;
	
	var classifierSelect = document.getElementById("classifierSelect");
	var classifierValue  = classifierSelect.options[classifierSelect.selectedIndex].text;
	
	//var node = document.getElementById("classifylist");
	
	//var outputLabel = document.createTextNode(featureString);
	//outputLabel.name = "outputLabel";
	
	//var outputLabel2 = document.createTextNode(path);
	//outputLabel2.name = "outputLabel2";
	
	//node.appendChild(outputLabel);
	//node.appendChild(outputLabel2);
	////app.onClassifyClick(event);
	
	//*************************CODE TO REQUEST AVAILABLE CLASSIFIERS****************
	
	//var classifiers_request = new XMLHttpRequest();
	//classifiers_request.open('GET', '/receiveavailableclassifiers', true);
	
	//classifiers_request.onreadystatechange=function() {
	//	if (classifiers_request.readyState==4)
	//	{
	//		if (classifiers_request.status>=200 && classifiers_request.status<300)
	//		{
	//			var res = JSON.parse(classifiers_request.response);
	//			for (var i = 0; i < res.Classifiers.length; i++) { 
	//			    alert(res.Classifiers[i].name);
	//			}
	// 		}
	//		else
	//		{
	//			var res = JSON.parse(classifiers_request.response);
	//			alert("Error: " + res.error);
	//		}
	//	}
	// }
    
	//classifiers_request.send();
    
	
	//***********************************************************************
	
	var path = dwv.tool.tools.draw.getPath();
	if (path.length === 0)
	{
		var path = dwv.tool.tools.livewire.getPath().pointArray;
	}
//	alert(path);
	var fileselect = document.getElementById("imagefiles");
	var featureString = '[' + featurevalue1 + ','  + featurevalue2 + ',' + featurevalue3 + ',' + featurevalue4 + ',' + featurevalue5 + ',' + featurevalue6 + ',' + featurevalue7 + ',' + featurevalue8 + ']';
	
	$("#popupClassify").popup("close");
	
	if(path.length === 0){
		alert('Please use the ROI (under Draw) or livewire tool to annotate a nodule before classification.');
	    return;
	}
/*	if(fileselect.files.length === 0){
		alert('Please load a DICOM image and use the livewire tool to annotate a nodule before classification.');
	    return;
	}*/
	
	var file = document.getElementById("imagefiles").files[0];
//	alert(file);
    var fd = new FormData();
	fd.append('upload', file);
	fd.append('path', path);
	fd.append('subjective', featureString);
	fd.append('classifier', classifierValue);
	
    document.getElementById("progress_div").style.display='block';
    document.getElementById("classifybutton").disabled = true;
    
	var classify_request = new XMLHttpRequest();
	
    classify_request.onreadystatechange=function()
    {
    //classify_request.onload=function()
    //{
    	if (classify_request.readyState == 4)
    	{
    		document.getElementById("progress_div").style.display='none';
    		document.getElementById("classifybutton").disabled = false;
    		if (classify_request.status >= 200 && classify_request.status < 300)
    		{
    			if (classifierValue == 'LIDC')
    			{
    				var res = JSON.parse(classify_request.response);
    				alert("Objective Malignancy: " + res.objective_result + "\nSubjective Malignancy: " + res.subjective_result);
    			}
    			else
    			{
    				var res = JSON.parse(classify_request.response);
    				alert("Objective Malignancy: " + res.objective_result);
    			}
    		}
    		else
    		{
    			var res = JSON.parse(classify_request.response);
    			alert("Error: " + res.error);
    		}
    	}
    }
    
    classify_request.onerror=function()
    {
    	document.getElementById("progress_div").style.display='none';
    	document.getElementById("classifybutton").disabled = false;
	
		alert("There was an error contacting the server");
    }
    
	classify_request.open('POST', '/sendall', true);
	classify_request.timeout = 20 * 60 * 1000; // 20 min timeout!
	
	classify_request.ontimeout = function () { $("#popupClassify").popup("close"); alert("The processing request timed out"); }
    
    classify_request.send(fd);
};


/**
 * Handle version request.
 * @method onClassifyClick
 * @static
 * @return {String} The version of the application.
 */
dwv.gui.versionRequest = function()
{
	//*************************CODE TO REQUEST VERSION****************
	var version_request = new XMLHttpRequest();
	version_request.open('GET', '/receiveversionnumber', false);
    
	version_request.send(null);
	
	if (version_request.readyState==4)
	{
		if (version_request.status>=200 && version_request.status<300)
		{
			return JSON.parse(version_request.response).version;
 		}
		else
		{
			return JSON.parse(classifiers_request.response).error;
		}
	}
};



/**
 * Handle files change.
 * @method onChangeFiles
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onChangeFiles = function(event)
{
    app.onChangeFiles(event);
};

/**
 * Handle URL change.
 * @method onChangeURL
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onChangeURL = function(event)
{
    app.onChangeURL(event);
};

/**
 * Handle tool change.
 * @method onChangeTool
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onChangeTool = function(/*event*/)
{
    app.getToolBox().setSelectedTool(this.value);
};

/**
 * Handle filter change.
 * @method onChangeFilter
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onChangeFilter = function(/*event*/)
{
    app.getToolBox().getSelectedTool().setSelectedFilter(this.value);
};

/**
 * Handle filter run.
 * @method onRunFilter
 * @static
 * @param {Object} event The run event.
 */
dwv.gui.onRunFilter = function(/*event*/)
{
    app.getToolBox().getSelectedTool().getSelectedFilter().run();
};

/**
 * Handle min/max slider change.
 * @method onChangeMinMax
 * @static
 * @param {Object} range The new range of the data.
 */
dwv.gui.onChangeMinMax = function(range)
{
    // seems like jquery is checking if the method exists before it 
    // is used...
    if( app.getToolBox().getSelectedTool().getSelectedFilter ) {
        app.getToolBox().getSelectedTool().getSelectedFilter().run(range);
    }
};

/**
 * Handle shape change.
 * @method onChangeShape
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onChangeShape = function(/*event*/)
{
    app.getToolBox().getSelectedTool().setShapeName(this.value);
};

/**
 * Handle line color change.
 * @method onChangeLineColour
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onChangeLineColour = function(/*event*/)
{
    app.getToolBox().getSelectedTool().setLineColour(this.value);
};

/**
 * Handle zoom reset.
 * @method onZoomReset
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onZoomReset = function(/*event*/)
{
    app.getImageLayer().resetLayout();
    app.getImageLayer().draw();
    if( app.getDrawLayer() ) {
        app.getDrawLayer().resetLayout();
        app.getDrawLayer().draw();
    }
};

/**
 * Handle display reset.
 * @method onDisplayReset
 * @static
 * @param {Object} event The change event.
 */
dwv.gui.onDisplayReset = function(event)
{
    dwv.gui.onZoomReset(event);
    app.initWLDisplay();
    // update preset select
    var select = document.getElementById("presetSelect");
    select.selectedIndex = 0;
    dwv.gui.refreshSelect("#presetSelect");
};

/**
 * Handle undo.
 * @method onUndo
 * @static
 * @param {Object} event The associated event.
 */
dwv.gui.onUndo = function(/*event*/)
{
    app.getUndoStack().undo();
};

/**
 * Handle redo.
 * @method onRedo
 * @static
 * @param {Object} event The associated event.
 */
dwv.gui.onRedo = function(/*event*/)
{
    app.getUndoStack().redo();
};

/**
 * Handle toggle of info layer.
 * @method onToggleInfoLayer
 * @static
 * @param {Object} event The associated event.
 */
dwv.gui.onToggleInfoLayer = function(/*event*/)
{
    app.toggleInfoLayerDisplay();
};
