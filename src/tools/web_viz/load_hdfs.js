var fileLocationURL = '/home/antonk/hello.txt';
var WebHDFS = require('webhdfs');
var hdfs = WebHDFS.createClient({
    user: 'antonk',
      host: '127.0.0.1',
        port: '50070',
          path: '/data/hello.txt'
});

readHDFSFile(fileLocationURL);


function readHDFSFile(locationOfHDFSFile){

  var remoteFileStream = hdfs.createReadStream(locationOfHDFSFile);

  remoteFileStream.on('error', function onError (err) {
      // Do something with the error
      console.log(err);
  });
      
  remoteFileStream.on('data', function onChunk (chunk) {
      // Do something with the data chunk
      console.log(chunk.toString());
  });
      
  remoteFileStream.on('finish', function onFinish () {
      // Upload is done
  });
      
}
