"use strict";

var canvas;
var gl;

var points = [];
var colors = [];

var xAxis = 0;
var yAxis = 1;
var zAxis = 2;

var TxAxis = 0;
var TyAxis = 1;
var TzAxis = 2;

var SxAxis = 0;
var SyAxis = 1;
var SzAxis = 2;

var axis = 0;
var Taxis = 0;
var Saxis = 0;
var theta = [0, 0, 0];
var Ttheta = [0, 0, 0];
var Stheta = [0, 0, 0];

var thetaLoc;
var TthetaLoc;
var SthetaLoc;
var flag = 0;
var flag1 = 0;

window.onload = function initCube() {
    canvas = document.getElementById("rtcb-canvas");

    gl = WebGLUtils.setupWebGL(canvas);
    if (!gl) {
        alert("WebGL isn't available");
    }

    makeCube();

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(1.0, 1.0, 1.0, 1.0);

    gl.enable(gl.DEPTH_TEST);

    // load shaders and initialize attribute buffer
    var program = initShaders(gl, "rtvshader", "rtfshader");
    gl.useProgram(program);

    var cBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

    var vColor = gl.getAttribLocation(program, "vColor");
    gl.vertexAttribPointer(vColor, 4, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vColor);

    var vBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points), gl.STATIC_DRAW);

    var vPosition = gl.getAttribLocation(program, "vPosition");
    gl.vertexAttribPointer(vPosition, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vPosition);

    thetaLoc = gl.getUniformLocation(program, "theta");
    gl.uniform3fv(thetaLoc, theta);

    TthetaLoc = gl.getUniformLocation(program, "Ttheta");
    gl.uniform3fv(TthetaLoc, Ttheta);

    SthetaLoc = gl.getUniformLocation(program, "Stheta");
    gl.uniform3fv(SthetaLoc, Stheta);

    document.getElementById("xbutton").onclick = function () {
        axis = xAxis;
        theta = [0, 0, 0];
        Ttheta = [0, 0, 0];
        Stheta = [0, 0, 0];
    }

    document.getElementById("ybutton").onclick = function () {
        axis = yAxis;
        theta = [0, 0, 0];
        Ttheta = [0, 0, 0];
        Stheta = [0, 0, 0];
    }

    document.getElementById("zbutton").onclick = function () {
        axis = zAxis;
        theta = [0, 0, 0];
        Ttheta = [0, 0, 0];
        Stheta = [0, 0, 0];
    }
    document.getElementById("xtrans").onclick = function () {
        Taxis = TxAxis;
        theta = [0, 0, 0];
        Ttheta = [0, 0, 0];
        Stheta = [0, 0, 0];
        flag=1;
    }

    document.getElementById("ytrans").onclick = function () {
        Taxis = TyAxis;
        theta = [0, 0, 0];
        Ttheta = [0, 0, 0];
        Stheta = [0, 0, 0];
    }

    document.getElementById("ztrans").onclick = function () {
        Taxis = TzAxis;
        theta = [0, 0, 0];
        Ttheta = [0, 0, 0];
        Stheta = [0, 0, 0];
    }

    document.getElementById("xscale").onclick = function () {
        Saxis = SxAxis;
        theta = [0, 0, 0];
        Ttheta = [0, 0, 0];
        Stheta = [0, 0, 0];
        flag1=1;
    }

    document.getElementById("yscale").onclick = function () {
        Saxis = SyAxis;
        theta = [0, 0, 0];
        Ttheta = [0, 0, 0];
        Stheta = [0, 0, 0];
    }

    document.getElementById("zscale").onclick = function () {
        Saxis = SzAxis;
        theta = [0, 0, 0];
        Ttheta = [0, 0, 0];
        Stheta = [0, 0, 0];
    }
    render();
    
}

function makeCube() {
    var vertices = [
        glMatrix.vec4.fromValues(-0.5, -0.5, 0.5, 1.0),
        glMatrix.vec4.fromValues(-0.5, 0.5, 0.5, 1.0),
        glMatrix.vec4.fromValues(0.5, 0.5, 0.5, 1.0),
        glMatrix.vec4.fromValues(0.5, -0.5, 0.5, 1.0),
        glMatrix.vec4.fromValues(-0.5, -0.5, -0.5, 1.0),
        glMatrix.vec4.fromValues(-0.5, 0.5, -0.5, 1.0),
        glMatrix.vec4.fromValues(0.5, 0.5, -0.5, 1.0),
        glMatrix.vec4.fromValues(0.5, -0.5, -0.5, 1.0),
    ];

    var vertexColors = [
        glMatrix.vec4.fromValues(0.0, 0.0, 0.0, 1.0),
        glMatrix.vec4.fromValues(1.0, 0.0, 0.0, 1.0),
        glMatrix.vec4.fromValues(1.0, 1.0, 0.0, 1.0),
        glMatrix.vec4.fromValues(0.0, 1.0, 0.0, 1.0),
        glMatrix.vec4.fromValues(0.0, 0.0, 1.0, 1.0),
        glMatrix.vec4.fromValues(1.0, 0.0, 1.0, 1.0),
        glMatrix.vec4.fromValues(0.0, 1.0, 1.0, 1.0),
        glMatrix.vec4.fromValues(1.0, 1.0, 1.0, 1.0)
    ];

    var faces = [
        1, 0, 3, 1, 3, 2, //正
        2, 3, 7, 2, 7, 6, //右
        3, 0, 4, 3, 4, 7, //底
        6, 5, 1, 6, 1, 2, //顶
        4, 5, 6, 4, 6, 7, //背
        5, 4, 0, 5, 0, 1  //左
    ];

    for (var i = 0; i < faces.length; i++) {
        points.push(vertices[faces[i]][0], vertices[faces[i]][1], vertices[faces[i]][2]);

        colors.push(vertexColors[Math.floor(i / 6)][0], vertexColors[Math.floor(i / 6)][1], vertexColors[Math.floor(i / 6)][2], vertexColors[Math.floor(i / 6)][3]);
    }
}

function render() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    theta[axis] += 0.1;
    if(flag==1){
        if(Ttheta[Taxis] < 1)
    {Ttheta[Taxis] += 0.001;}
	else{Theta[Taxis] = 0;} 
    }
    if(flag1==1){
        if(Stheta[Saxis] < 2){Stheta[Saxis] += 0.001;}
		else{Stheta[Saxis] = 0;}	
    }
    
    gl.uniform3fv(thetaLoc, theta);
    gl.uniform3fv(TthetaLoc, Ttheta);
    gl.uniform3fv(SthetaLoc, Stheta);

    gl.drawArrays(gl.TRIANGLES, 0, points.length / 3);

    requestAnimFrame(render);
}

