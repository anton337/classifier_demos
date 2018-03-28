// Include created client
var hdfs = require('./webhdfs-client');

// Include fs module for local file system operations
var fs = require('fs');

// Initialize readable stream from local file
// Change this to real path in your file system
var localFileStream = fs.createReadStream('/path/to/local/file');

// Initialize writable stream to HDFS target
var remoteFileStream = hdfs.createWriteStream('/path/to/remote/file');

// Pipe data to HDFS
localFileStream.pipe(remoteFileStream);

// Handle errors
remoteFileStream.on('error', function onError (err) {
  // Do something with the error
});

// Handle finish event
remoteFileStream.on('finish', function onFinish () {
  // Upload is done
});
