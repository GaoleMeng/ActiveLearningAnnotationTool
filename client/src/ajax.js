// -*- Mode: JavaScript; tab-width: 2; indent-tabs-mode: nil; -*-
// vim:set ft=javascript ts=2 sw=2 sts=2 cindent:
var Ajax = (function($, window, undefined) {
    var PROTOCOL_VERSION = 1
    var Ajax = function(dispatcher) {
      var that = this;
      var pending = 0;
      var count = 0;
      var pendingList = {};

      // merge data will get merged into the response data
      // before calling the callback
      var ajaxCall = function(data, callback, merge, extraOptions) {
        merge = merge || {};
        dispatcher.post('spin');
        pending++;
        var id = count++;

        // special value: `merge.keep = true` prevents obsolescence
        pendingList[id] = merge.keep || false;
        delete merge.keep;

        // If no protocol version is explicitly set, set it to current
        if (data.toString() == '[object FormData]') {
          data.append('protocol', PROTOCOL_VERSION);
        } else if (data['protocol'] === undefined) {
          // TODO: Extract the protocol version somewhere global
          data['protocol'] = PROTOCOL_VERSION;
        }

        options = {
            url: 'ajax.cgi',
            data: data,
            type: 'POST',
            success: function(response) {
              pending--;
              // If no exception is set, verify the server results
              if (response.exception == undefined && response.action !== data.action) {
                console.error('Action ' + data.action +
                  ' returned the results of action ' + response.action);
                response.exception = true;
                dispatcher.post('messages', [[['Protocol error: Action' + data.action + ' returned the results of action ' + response.action + ' maybe the server is unable to run, please run tools/troubleshooting.sh from your installation to diagnose it', 'error', -1]]]);
              }


              // console.log(response.action);


              //retrain model demonstration:
              if (response.action == "retrainmodel"){

                  location.reload();
                  //dispatcher.post('messages', [[['Backend model trained', 'error']]]);

                  // console.log($(".background > rect:first").attr("x"));
                  // function readTextFile(file)
                  // {
                  //     var rawFile = new XMLHttpRequest();
                  //     rawFile.open("GET", file, false);
                  //     rawFile.onreadystatechange = function ()
                  //     {
                  //         if(rawFile.readyState === 4)
                  //         {
                  //             if(rawFile.status === 200 || rawFile.status == 0)
                  //             {
                  //                 var allText = rawFile.responseText;

                  //                 var lines = allText.split("\n");
                  //                 var toggle = 1;
                  //                 var startpos = 0;
                  //                 // console.log($(".background > rect"));
                  //                 for (var i = 0; i < lines.length; i++){
                  //                     if (toggle){
                  //                       toggle = 0;
                  //                     }
                  //                     else{
                  //                       toggle = 1;
                  //                     }
                  //                     if (lines[i].length != 0){
                  //                         topvalue = $(".background > rect:eq("+startpos+")").position();
                  //                         console.log(topvalue)
                  //                         // inttop = parseInt(topvalue);
                  //                         // console.log(inttop);
                  //                         var fatherdiv = document.getElementById("classifier");
                  //                         var childdiv = document.createElement("div");
                  //                         var line = document.createElement("hr")
                  //                         childdiv.innerHTML = lines[i];
                  //                         childdiv.style.position = "absolute";
                  //                         childdiv.style.top = topvalue.top+"px";
                  //                         line.style.position = "absolute";
                  //                         line.style.top = topvalue.top+"px";
                  //                         // childdiv.id = "child"+i;
                  //                         // $("#child"+i).style.top = "100px";
                  //                         $("#classfifier").append(line);
                  //                         $("#classfifier").append(childdiv);
                  //                     }
                  //                     console.log($(".background > rect:eq("+startpos+")").attr("class"), "background"+toggle);

                  //                     while (startpos < $(".background > rect").length && $(".background > rect:eq("+startpos+")").attr("class")==("background"+toggle)){
                  //                         startpos++;
                  //                     }
                  //                 }
                  //             }
                  //         }
                  //     }
                  //     rawFile.send(null);
                  //}

                  //readTextFile("test.txt");

              }


              // If the request is obsolete, do nothing; if not...
              if (pendingList.hasOwnProperty(id)) {
                dispatcher.post('messages', [response.messages]);
                if (response.exception == 'configurationError'
                    || response.exception == 'protocolVersionMismatch') {
                  // this is a no-rescue critical failure.
                  // Stop *everything*.
                  pendingList = {};
                  dispatcher.post('screamingHalt');
                  // If we had a protocol mismatch, prompt the user for a reload
                  if (response.exception == 'protocolVersionMismatch') {
                    if(confirm('The server is running a different version ' +
                        'from brat than your client, possibly due to a ' +
                        'server upgrade. Would you like to reload the ' +
                        'current page to update your client to the latest ' +
                        'version?')) {
                      window.location.reload(true);
                    } else {
                      dispatcher.post('messages', [[['Fatal Error: Protocol ' +
                          'version mismatch, please contact the administrator',
                          'error', -1]]]);
                    }
                  }
                  return;
                }

                delete pendingList[id];

                // if .exception is just Boolean true, do not process
                // the callback; if it is anything else, the
                // callback is responsible for handling it
                if (response.exception == true) {
                  $('#waiter').dialog('close');
                } else if (callback) {
                  $.extend(response, merge);
                  dispatcher.post(0, callback, [response]);
                }


              }
              dispatcher.post('unspin');
            },
            error: function(response, textStatus, errorThrown) {
              pending--;
              dispatcher.post('unspin');
              $('#waiter').dialog('close');
              dispatcher.post('messages', [[['Error: Action' + data.action + ' failed on error ' + response.statusText, 'error']]]);
              console.error(textStatus + ':', errorThrown, response);
            }
          };

        if (extraOptions) {
          $.extend(options, extraOptions);
        }
        $.ajax(options);
        return id;
      };

      var isReloadOkay = function() {
        // do not reload while data is pending
        return pending == 0;
      };

      var makeObsolete = function(all) {
        if (all) {
          pendingList = {};
        } else {
          $.each(pendingList, function(id, keep) {
            if (!keep) delete pendingList[id];
          });
        }
      }

      dispatcher.
          on('isReloadOkay', isReloadOkay).
          on('makeAjaxObsolete', makeObsolete).
          on('ajax', ajaxCall);
    };

    return Ajax;
})(jQuery, window);
