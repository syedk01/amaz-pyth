var request    = require('request');
var sys        = require('sys');
var server     = require('http');
var qs         = require('querystring');
var multipart  = require('multipart');
var readFile   = require('fs').readFile;
var deleteFile = require('fs').unlink;
var url        = require('url');
var exec       = require('child_process').exec;
var path       = require('path');
var fs         = require('fs');
var formidable = require('formidable');
var util       = require('util');
var connect    = require('connect');
var serveStatic= require('serve-static');
var rand       = require('generate-key');
   	
var toolbox_path     = '/home/ubuntu';
//var result_path      = toolbox_path + fs.readFileSync(toolbox_path + '/output_path.txt', "utf8").replace('.','') + '/';
//var upload_path      = toolbox_path + fs.readFileSync(toolbox_path + '/dataset_path_LIDC.txt', "utf8").replace('.','') + '/';
var result_path      = toolbox_path + '/OUTPUT/';
var upload_path      = toolbox_path + '/FILE/';
var html_base_path   = './site';
var version_number   = fs.readFileSync('version_number.txt', "utf8");

var classifer_type='svm';

if( process.argv[2]==1)
{
	classifer_type='knn';
}
else if( process.argv[2]==2 )
{
	classifer_type='nnet';
}

else
{
	classifer_type='svm';
}
//var local_processing  = process.argv[2];
//var matlab_processing = process.argv[3];

sys.puts('*************************************************************************************');
sys.puts(' Pacific Cloud Nodule Classification GUI Server Version ' + version_number);
sys.puts(' Using ' + upload_path + ' to store the uploaded images');
sys.puts(' Using ' + result_path + ' to store the results of the computation');
/* if (local_processing === '1')
{
	sys.puts('Processing on local server');
}
else
{
	sys.puts('Processing on Domino\'s servers');
}
if (matlab_processing === '1')
{
	sys.puts('Using Matlab');
	var processing_platform = 'matlab';
}
else
{
	sys.puts('Using Octave');
	var processing_platform = 'octave';
}
sys.puts('*************************************************************************************'); */



/*
 * Create the server
 */
server.createServer(function(req, res){
	
	switch (url.parse(req.url).pathname) {
	//	case '/sendimage':
	//	sys.puts('Receiving image...');
	//	upload_file(req, res);
	//	break;
	//case '/sendfeatures':
	//	sys.puts('Receiving features...');
	//	receive_features(req, res);
	//	break;
	case '/sendall':
		//var date = new Date();
		//sys.puts(date.getHours()+":"+date.getMinutes()+":"+date.getSeconds() + ' ' + req.url);
		//console.log(req.headers);
		sys.puts('Image, path, and subjective features send request...');
		receive_all(req, res);
		break;
	case '/receiveavailableclassifiers':
		sys.puts('Available classifiers request...');
		send_available_classifiers(req, res);
		break;
	case '/receiveversionnumber':
		sys.puts('Version number request...');
		send_version_number(req, res);
		break;
	case '/loadfromurl':
		sys.puts('Loading from url...');
		load_url(req, res);
		break;
	default:
		serve_webpage(req, res);
		break;
	}
}).on('connection', function(socket) {
	socket.setTimeout(20 * 60 * 1000); // set to a 20 min timeout
}).listen(8080);



/*
 * Handle objective classification task
 */


function upload_file(req, res) {

	var form = new formidable.IncomingForm();

	form.keepExtensions = true;
	form.uploadDir = upload_path;

	form.parse(req, 
		function(err, fields, files)
		{

			if (files.upload == undefined)
			{
				return_error(req, res, 'Uploaded object does not contain an upload file', 'Uploaded object does not contain an upload file');
				return;
			}
			if (fields.path == undefined)
			{
				return_error(req, res, 'Uploaded object does not contain a path field', 'Uploaded object does not contain a path field');
				return;
			}

			//console.log('\tInput Filename: ' + sys.inspect({fields: fields, files: files}));
			console.log('\tInput filename: ' + files.upload.name);
			console.log('\tOutput filename: ' + files.upload.path);

			fs.writeFile(files.upload.name, files.upload, 'utf8', 
				function (err)
				{
					if (err)
					{
						return_error(req, res, 'Could not write image file', 'Server could not write image file');
						return;
					}
				}
			);
    
			// The file is cached locally under the original name so delete this
			if (fs.existsSync(files.upload.name))
			{
				deleteFile(files.upload.name);
			}
			
    
			if (local_processing == 1)
			{
				if (process.platform === 'win32')
				{
					var command = 'local_classify_objective_' + processing_platform + '.bat';
				}
				else
				{
					var command = './local_classify_objective_' + processing_platform + '.sh'
				}
				var execute_command = command + ' \"' + files.upload.path + '\" \"' + fields.path + '\"';
			}
			else
			{
				sys.puts('Writing classify_launcher script: ' + 'classify_nodule_objective \'' + files.upload.path + '\' \'' + fields.path + '\'');
				fs.writeFileSync(toolbox_path + '/classify_launcher_objective.m', 'classify_nodule_objective \'' + files.upload.path + '\' \'' + fields.path + '\'', 'utf8');
				if (process.platform === 'win32'){
					var execute_command = './domino_classify_objective_' + processing_platform + '.bat';
				}
				else
				{
					var execute_command = './domino_classify_objective_' + processing_platform + '.sh'
				}
			}
			
			sys.puts('Executing: ' + execute_command);
			exec(execute_command, 
				function (error, stdout, stderr)
				{
					sys.puts('\tstdout: ' + stdout);
					sys.puts('\tstderr: ' + stderr);
					if (error !== null)
					{
						sys.puts('\texec error: ' + error);
					}
                 
					deleteFile(files.upload.path);
                 
                    // Name the result file as the saved filename plus _results.txt
                    result_file_obj = files.upload.path.substring(files.upload.path.lastIndexOf(path.sep)+1, files.upload.path.length).split('.');
                    result_file_obj = result_path + result_file_obj[0] + '_results_obj.txt';
                 
					sys.puts('Reading result file: ' + result_file_obj);
					readFile(result_file_obj, 'utf8', 
							function(err, data)
							{
								if (err)
								{
									return_error(req, res, 'Result file not found, command probably didn\'t execute properly', 'Error executing computation on server');
				                 	return;
								}
		     
								sys.puts('\t' + data);
								deleteFile(result_file_obj);
		     
								var response_text = JSON.stringify({'result': data});
								res.writeHead(200, {'content-type': 'application/json'});
								res.write(response_text);
								res.end();
                     
								sys.puts('Response sent: ' + response_text);
							}
					);
				}
			);
		}
	);
}

function test(req, res) {
	var response_text = JSON.stringify({'subjective_result': '0.1', 'objective_result': '0.1'});
	res.writeHead(200, {'content-type': 'application/json'});
	res.write(response_text);
	res.end();
}

/*
 * Handle Objective and Subjective classification task
 */
function receive_all(req, res) {
	
	var form = new formidable.IncomingForm();

	form.keepExtensions = true;
	form.uploadDir = upload_path;

	form.parse(req, 
		function(err, fields, files)
		{

			if (files.upload == undefined)
			{
				return_error(req, res, 'Uploaded object does not contain an upload file', 'Uploaded object does not contain an upload file');
				return;
			}
			if (fields.path == undefined)
			{
				return_error(req, res, 'Uploaded object does not contain a path field', 'Uploaded object does not contain a path field');
				return;
			}
			if (fields.subjective == undefined)
			{
				return_error(req, res, 'Uploaded object does not contain a subjective field', 'Uploaded object does not contain a subjective field');
				return;
			}
			if (fields.classifier == undefined)
			{
				return_error(req, res, 'Uploaded object does not contain an classifier field', 'Uploaded object does not contain an classifier field');
				return;
			}

			//console.log('\tInput Filename: ' + sys.inspect({fields: fields, files: files}));
			console.log('\tInput filename: ' + files.upload.name);
			console.log('\tOutput filename: ' + files.upload.path);
            
			fs.writeFile(files.upload.name, files.upload, 'utf8', 
				function (err)
				{
					if (err)
					{
						return_error(req, res, 'Could not write image file', 'Server could not write image file');
						return;
					}
				}
			);
			
			// The file is cached locally under the original name so delete this
			if (fs.existsSync)
			{
				deleteFile(files.upload.name);
			}
    
    
			/* if (local_processing == 1)
			{
				if (process.platform === 'win32')
				{
					var command = 'local_classify_nodule_' + processing_platform + '.bat';
				}
				else
				{
					var command = './local_classify_nodule_' + processing_platform + '.sh'
				}
				
				var execution_command = command + ' \"' + fields.classifier + '\" \"' + files.upload.path + '\" \"' + fields.path + '\" \"' + fields.subjective + '\"';
			}
			else
			{
				sys.puts('Writing classify_launcher script: ' + 'classify_nodule \'' + fields.classifier + '\' \'' + files.upload.path + '\' \'' + fields.path + '\' \'' + fields.subjective + '\'');
				fs.writeFileSync(toolbox_path + '/classify_launcher.m', 'classify_nodule \'' + fields.classifier + '\' \'' + files.upload.path + '\' \'' + fields.path + '\' \'' + fields.subjective + '\'', 'utf8');
				if (process.platform === 'win32'){
					var execution_command = 'domino_classify_nodule_' + processing_platform + '.bat';
				}
				else
				{
					var execution_command = './domino_classify_nodule_' + processing_platform + '.sh'
				}
			} */
			
			var result_file = files.upload.path.substring(files.upload.path.lastIndexOf(path.sep)+1, files.upload.path.length).split('.');
            result_file = result_path + result_file[0] + '_results.txt';
			
			var execution_command='sudo '+toolbox_path+'/run_classifier.py '+ classifer_type + ' ' + fields.classifier + ' ' + fields.subjective + ' ' +  result_file + ' ' + files.upload.path + ' ' + '\''+fields.path+'\'' 
			sys.puts('Executing: ' + execution_command);
			exec(execution_command, 
				function (error, stdout, stderr)
				{
					sys.puts('\tstdout: ' + stdout);
					sys.puts('\tstderr: ' + stderr);
					if (error !== null)
					{
						sys.puts('\texec error: ' + error);
					}
					
					deleteFile(files.upload.path);
                 
                    // Name the result file as the saved filename plus _results.txt
                    result_file = files.upload.path.substring(files.upload.path.lastIndexOf(path.sep)+1, files.upload.path.length).split('.');
                    result_file = result_path + result_file[0] + '_results.txt';
                 
					sys.puts('Reading result file: ' + result_file);
					readFile(result_file, 'utf8', 
							function(err, data)
							{
								if (err)
								{
									return_error(req, res, 'Result file not found, command probably didn\'t execute properly', 'Error executing computation on server');
				                 	return;
								}
		     
								data = data.split(',');
								
								sys.puts('\tObjective Result: '  + data[0]);
								sys.puts('\tSubjective Result: ' + data[1]);
								deleteFile(result_file);
		     
								var response_text = JSON.stringify({'subjective_result': data[1], 'objective_result': data[0]});
								res.writeHead(200, {'content-type': 'application/json'});
								res.write(response_text);
								res.end();
                     
								sys.puts('Response sent: ' + response_text);
							}
					);
				}
			);
		}
	);
}


/*
 * Receive subjective features
 */
function receive_features(req, res){
	if (req.method == 'POST')
	{
		var body = '';

		req.on('data', 
			function(data)
			{
				body += data;
        	}
		);
        
		req.on('end', 
			function()
			{
        		var input = JSON.parse(body);

            	if (input.features == undefined)
            	{
            		return_error(req, res, 'Uploaded object does not contain a features field', 'Uploaded object does not contain a features field');
                 	return;
            	}

            			sys.puts('Received POST body - features: ' + input.features);
            
            			input.features = input.features.replace(/\s+/g, ''); // Remove spaces in feature vector

            			sys.puts('Received the following features: ' + input.features);
            			sys.puts('Received the following path: '     + input.path);
            

            			if (local_processing == 1)
            			{
                			if (process.platform === 'win32')
                			{
                    			var command = 'local_classify_subjective.bat';
                			}
                			else
                			{
                    			var command = './local_classify_subjective.sh'
                			}
                			var execute_command = command + ' \"' + input.features + '\"';
            			}
            			else
            			{
            				sys.puts('Writing classify_launcher script: ' + 'classify_nodule_subjective \'' + fields.subjective + '\'');
            				fs.writeFileSync(toolbox_path + '/classify_launcher.m', 'classify_nodule_subjective \'' + fields.subjective + '\'', 'utf8');
            				if (process.platform === 'win32')
            				{
                    			var execute_command = 'domino_classify_subjective.bat';
            				}
            				else
            				{
            					var execute_command = './domino_classify_subjective.sh'
            				}
            			}

				sys.puts('Executing: ' + execute_command);
				exec(execute_command,
					function (error, stdout, stderr)
					{
						sys.puts('\tstdout: ' + stdout);
						sys.puts('\tstderr: ' + stderr);
						if (error !== null)
						{
							sys.puts('\texec error: ' + error);
						}

                     
                        // Name the result file as the saved filename plus _results.txt
                        result_file_sub = files.upload.path.substring(files.upload.path.lastIndexOf(path.sep)+1, files.upload.path.length).split('.');
                        result_file_sub = result_path + result_file_sub[0] + '_results_sub.txt';
                     
						sys.puts('Reading result file: ' + result_file_sub);
						readFile(result_file_sub, 'utf8', 
								function(err, data)
								{

									if (err)
									{
										return_error(req, res, 'Result file not found, command probably didn\'t execute properly', 'Error executing computation on server');
										return;
									}

									sys.puts('\t'+data);
									deleteFile(result_file_sub);
				
									var response_text = JSON.stringify({'result': data});
									res.writeHead(200, {'content-type': 'application/json'});
									res.write(response_text);
									res.end();

									sys.puts('Response sent: ' + response_text);
								}
						);
					}
				);
			}
		);
	}
}


/*
 * Handles page not found error
 */
function show_404(req, res)
{
	res.statusCode = 404;
	res.setHeader('Content-Type', 'text/html');
	res.write('Page Not Found');
	res.end();
}

/*
 * 
 */
function return_error(req, res, local_err_msg, return_err_msg)
{
	sys.puts(local_err_msg);
 	 
	res.writeHead(400, {'content-type': 'application/json'});
	res.write(JSON.stringify({'error': return_err_msg}));
	res.end();
	
	sys.puts('Error sent');
}


/*
 * 
 */
function send_available_classifiers(req, res)
{
	if (req.method === 'GET')
	{
		if (matlab_processing === '1')
			var data = {'Classifiers': [{'name':'nn'},{'name':'svm'},{'name':'knn'}]};
		else
			var data = {'Classifiers': [{'name':'nn'},{'name':'svm'}]};
				
		var response_text = JSON.stringify(data);
		res.writeHead(200, {'content-type': 'application/json'});
		res.write(response_text);
		res.end();
		
		sys.puts('Sent: ' + response_text);
	}
	else
	{
		return_error(req, res, 'Invalid request', 'Invalid request, use a GET');
	}
}


/*
 * 
 */
function send_version_number(req, res)
{
	if (req.method === 'GET')
	{
        version_file = 'version_number.txt';
     
		sys.puts('Reading version file: ' + version_file);
		readFile(version_file, 'utf8', 
				function(err, data)
				{
					if (err)
					{
						return_error(req, res, 'Version file not found', 'Version file not found on server');
						return;
					}

					sys.puts('\t' + data);
					
					var response_text = JSON.stringify({'version': data});
					res.writeHead(200, {'content-type': 'application/json'});
					res.write(response_text);
					res.end();

					sys.puts('Response sent: ' + response_text);
				}
		);
	}
	else
	{
		return_error(req, res, 'Invalid request', 'Invalid request, use a GET');
	}
}


/*
 * Serves the webpages
 */
function serve_webpage(req, res)
{
	var filePath = req.url;
	sys.puts(req.url);
	if (filePath == '/')
		filePath = path.sep + 'index.html';

	filePath = html_base_path + filePath;

	var extname = path.extname(filePath);
	var contentType = 'text/html';
	switch (extname)
	{
		case '.js':
			contentType = 'text/javascript';
			break;
		case '.css':
			contentType = 'text/css';
			break;
	}
	
	fs.exists(filePath, 
		function(exists)
		{
			if (exists) 
			{
				fs.readFile(filePath, 
					function(error, content)
					{
						if (error)
						{
							res.writeHead(500);
							res.end();
						}
						else
						{
							res.writeHead(200, { 'Content-Type': contentType });
							res.end(content, 'utf-8');
						}
					}
				);
			}
			else
			{
				show_404(req, res)
				console.log(filePath + ' does not exist!');
			}
		}
	);
}

function load_url(req, res) {
		//	sys.puts("body:");
			var body ='';
		//	var url='';
			//sys.puts(util.inspect(req));			//sys.puts(JSON.stringify(req));
			req.on('data', function(data){ 
				body +=data; 
				if (body.length> 1e6){
					req.connection.destroy();} 
				d = JSON.parse(body); 
			//	sys.puts("Data:"+ d.url+"\n Type:"+ typeof d.url); 
				var url = d.url;
		//	sys.puts(body);
		//	var url = req.on('end', function(){var post =  qs.parse(body); return post['url'];});
		//	var loadurl = JSON.parse(req.data);			
		//	var url = loadurl.url;
			var saveto = '/home/ubuntu/tmp/' + rand.generateKey() + ".dcm";
			var execution_command='wget -O ' + saveto+ ' ' + url;
			sys.puts('Executing: ' + execution_command);
			exec(execution_command, 
				function (error, stdout, stderr)
				{
					sys.puts('\tstdout: ' + stdout);
					sys.puts('\tstderr: ' + stderr);
					if (error !== null)
					{
						sys.puts('\texec error: ' + error);
					}
                 
					
					fs.readFile(saveto, 
							function(err, data)
							{
								if (err)
								{
									sys.puts(err);
									return_error(req, res, 'Result file not found, command probably didn\'t execute properly', 'Error executing computation on server');
				                 			return;
								}
		     
								//data = data.split(',');
								
							//	sys.puts('\tObjective Result: '  + data[0]);
							//	sys.puts('\tSubjective Result: ' + data[1]);
								
		     
								//var response_text = JSON.stringify({'subjective_result': data[1], 'objective_result': data[0]});
								res.writeHead(200, {'content-type': 'application/octet-stream'});
								res.write(data);
								res.end();
                     
								sys.puts('Response sent');
								deleteFile(saveto);
							}
					);
				}
			);
	});		
}
