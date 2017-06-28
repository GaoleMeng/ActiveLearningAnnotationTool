
# Annotation Tool for active learning #


## About ##


To fully visulize the process of active learning, we develop an annotation tool based on a current annotation system named "Brat", (https://github.com/nlplab/brat), the core of the application are the annotation part as the Brat tool, and we developed the annotation tool with the functionalbity to show the training result whenever the annotator finish one set of annotation. The model in the backend will keep track of the current label data and gives out the most "wanted" data to be labeled.


## License ##

brat itself is available under the permissive [MIT License][mit] but
incorporates software using a variety of open-source licenses, for details
please see see LICENSE.md.

[mit]:  http://opensource.org/licenses/MIT

## Install ##

    sh install.sh
    python standalone.py
    

