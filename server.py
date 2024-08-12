from flask import Flask, request, make_response, jsonify
import src.ipc as ipc

app = Flask(__name__)

gpu = ipc.ipc("./build/comp.dll")
taskList = {
    "MAT_ADD": gpu.mat_add,
    "MAT_MULT": gpu.mat_mult,
    "MAT_TRANSPOSE": gpu.transpose,
}


@app.route('/')
def helloworld():
    return "Hello World!"


@app.route('/ping')
def ping():
    return 'pong'


@app.route('/compute', methods=['GET'])
def calculate():
    # the basic tenet of this function is to return an output array, given two input arrays
    if not request.is_json:
        return make_response("No JSON present!", 415)
    data = request.get_json()
    if data.get('optype', None) is None:
        return make_response("No operation specified!", 400)

    optype = data['optype']
    buf1 = []
    buf2 = []
    w1, h1, w2, h2 = (0, 0, 0, 0)

    single_buffer_optypes = ['MAT_TRANSPOSE']
    if data['optype'] in single_buffer_optypes:
        if data.get('b1', None) is None:
            return make_response("No buffer in JSON!", 400)
        if len(data['b1']) == 0 or not isinstance(data['b1'][0], list):
            return make_response("Incorrect buffer configuration. Make sure buffers are a 2D array of values and populated.", 400)
        buf1 = data['b1']
        w1 = len(buf1)
        h1 = len(buf1[0]) if w1 > 0 else 0
    else:
        if data.get('b1', None) is None or data.get('b2', None) is None:
            return make_response("Not all buffers in JSON!", 400)
        if len(data['b1']) == 0 or not isinstance(data['b1'][0], list) or len(data['b2']) == 0 or not isinstance(data['b2'][0], list):
            return make_response("Incorrect buffer configuration. Make sure buffers are a 2D array of values and populated.", 400)
        buf1 = data['b1']
        buf2 = data['b2']
        w1, w2 = len(buf1), len(buf2)
        h1, h2 = len(buf1[0]) if w1 > 0 else 0, len(buf2[0]) if w2 > 0 else 0

    # now that all error checking has been done, we can get down to business implementing the CUDA API call.
    print(
        f"Processing operation: {optype} with data: \n{(w1, h1)}\n{(w2, h2)}")

    replyJson = {"optype": optype, "result": []}
    replyJson['result'] = taskList[optype](
        buf1) if optype in single_buffer_optypes else taskList[optype](buf1, buf2)

    return make_response(jsonify(replyJson), 200)
