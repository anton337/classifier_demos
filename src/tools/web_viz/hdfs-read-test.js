var hdfs = require('./webhdfs-client');
var fs = require('fs');

// Initialize readable stream from HDFS source
var remoteFileStream = hdfs.createReadStream('/data/hello.txt');

var dat = [];

remoteFileStream.on('error', function onError (err) {
  // Do something with the error
  console.log(err.toString());
});

remoteFileStream.on('data', function onChunk (chunk) {
  // Concat received data chunk
  //console.log(chunk.toString());
  var tmp = chunk.toString().split("\n");
  var arrLen = tmp.length;
  for(var i=0;i<arrLen;i++)
  {
    dat.push(tmp[i]);
  }
});

remoteFileStream.on('finish', function onFinish () {
  // Upload is done
  // Print received data
  console.log(dat);
});
