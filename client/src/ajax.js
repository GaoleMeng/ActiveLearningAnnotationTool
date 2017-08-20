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
              if (response.action == "getDocument"){


                $.getJSON( "label.json", function( data ) {
          
                  coll = response.collection;
                  doc = response.document;
                  var newpath = "data/" + coll + "output/" + doc + ".lbl";
                  console.log(doc);
                  var problist = [];

                  function readTextFile(file)
                          {
                              var rawFile = new XMLHttpRequest();
                              rawFile.open("GET", file, false);
                              rawFile.onreadystatechange = function ()
                              {
                                  if(rawFile.readyState === 4)
                                  {
                                      if(rawFile.status === 200 || rawFile.status == 0)
                                      {
                                          var allText = rawFile.responseText;

                                          var lines = allText.split("\n");
                                          // console.log($(".background > rect"));
                                          for (var i = 0; i < lines.length; i++){
                                              if (lines[i].length != 0){
                                                  var vec = lines[i].split("\t");
                                                  console.log(vec)
                                                  problist.push(parseFloat(vec[1]));
                                              }
                                          }
                                      }
                                  }
                              }
                              rawFile.send(null);
                          }
                  readTextFile(newpath)

                  renderdiv = $("#render")
                  renderdiv.empty();
                  for (var i = 0; i < data["num"]; i++){
                      var childdiv = document.createElement("div");
                      var checkbox = document.createElement("input")
                      var griddiv = document.createElement("div");
                      var input_group_div = document.createElement("div");
                      var first_span = document.createElement("span");
                      var second_span = document.createElement("span")
                      var third_span = document.createElement("span");

                      childdiv.className = "row justify-content-md-center"
                      griddiv.className = "col-lg-4"
                      input_group_div.className = "input-group"
                      first_span.className = "input-group-addon";
                      second_span.className = "input-group-addon";
                      third_span.className = "input-group-addon";

                      first_span.append(checkbox);
                      childdiv.append(griddiv);
                      input_group_div.append(first_span);
                      input_group_div.append(second_span);
                      input_group_div.append(third_span);

                      griddiv.append(input_group_div);
                      childdiv.append(griddiv);


                      second_span.textContent = data[i];
                      third_span.textContent = problist[i];

                      checkbox.type = "checkbox";
                      var hh = i+1;
                      checkbox.id = "label" + hh;
                      checkbox.style.float = "left";
                      checkbox.value = "off";
                      
                      checkbox.role = "button"

                      childdiv.style.margin="3px"

                      $("#render").append(childdiv);
                      
                  }
              });

              }

              if (response.action == "retrainmodel"){
                  location.reload();
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
