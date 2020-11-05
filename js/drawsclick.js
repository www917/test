"use strict";

var canvas;
var gl;

var triangleIndex = 0;
var Tmove = [];
var squareIndex = 0;
var Smove = [];
var cubeIndex = 0;
var Cmove = [];
var circleIndex = 0;
var sideNum;
var Rmove = [];
var rRmove = [];
var circleVertex = [];
var	circleColors = [];
// 缩放
var Stheta = [0, 0, 0];
var SthetaLoc;
var theta = 0.0;
var thetaLoc;
var Ctheta = [0, 0, 0];
var CthetaLoc;

var shape="ss";

var moveLoc;

var colors = [
	0.0, 0.0, 0.0, 1.0, // black
	1.0, 0.0, 0.0, 1.0 , // red
	1.0, 1.0, 0.0, 1.0 , // yellow
	0.0, 1.0, 0.0, 1.0 , // green
	0.0, 0.0, 1.0, 1.0 , // blue
	1.0, 0.0, 1.0, 1.0 , // magenta
	0.0, 1.0, 1.0, 1.0  // cyan
];

var ponits = [
    // 三角形
    0.0, 0.21, 0.0,
	-0.169, -0.09, 0.0,
    0.171, -0.09, 0.0,
    
    // 正方形
    0.0, 0.2, 0.0, 
	-0.2, 0.0, 0.0,
	0.2, 0.0, 0.0,
	0.0, -0.2, 0.0,
];

function initSquare(){
	canvas = document.getElementById( "dot-canvas" );
	gl = WebGLUtils.setupWebGL( canvas );
	if( !gl ){
		alert( "WebGL isn't available" );
	}
	sideNum = document.getElementById("slide").value;
	makeCube();
	gl.viewport( 0, 0, canvas.width, canvas.height );
	gl.clearColor( 0.5, 0.5, 0.5, 1.0 );
	gl.enable(gl.DEPTH_TEST);
	
	var program = initShaders( gl, "vertex-shader", "fragment-shader" );
	gl.useProgram( program );
	
	SthetaLoc = gl.getUniformLocation(program, "Stheta");
	CthetaLoc = gl.getUniformLocation(program, "Ctheta");
	thetaLoc = gl.getUniformLocation(program, "theta");
	moveLoc = gl.getUniformLocation(program, "move");
	recreate();

function recreate(){
	var vBuffer = gl.createBuffer(); //position
	gl.bindBuffer( gl.ARRAY_BUFFER, vBuffer );
    gl.bufferData( gl.ARRAY_BUFFER, new Float32Array(ponits.concat(circleVertex)), gl.STATIC_DRAW );

	var vPosition = gl.getAttribLocation( program, "vPosition" );
	gl.vertexAttribPointer( vPosition, 3, gl.FLOAT, false, 0, 0 );
    gl.enableVertexAttribArray( vPosition );

	var cBuffer = gl.createBuffer(); // color
	gl.bindBuffer( gl.ARRAY_BUFFER, cBuffer );
	gl.bufferData( gl.ARRAY_BUFFER, new Float32Array(colors.concat(circleColors)), gl.STATIC_DRAW );

	var vColor = gl.getAttribLocation( program, "vColor" );
	gl.vertexAttribPointer( vColor, 4, gl.FLOAT, false, 0, 0 );
	gl.enableVertexAttribArray( vColor );
}
canvas.addEventListener( "mousedown", function( event ){
	var rect = canvas.getBoundingClientRect();
	var cx = event.clientX - rect.left;
	var cy = event.clientY - rect.top; // offset
	var NCx = 2 * cx / canvas.width - 1;
	var NCy = 2 * (canvas.height - cy) / canvas.height - 1;
	if(shape == "trangle"){
		triangleCreate(NCx, NCy);	
	}else if(shape == "square"){
		squareCreate(NCx, NCy);
	}else if(shape == "3Dsquare"){
		cubeCreate(NCx, NCy);
	}else if(shape == "round"){
		Rmove.push([NCx, NCy]);
		rRmove.push([NCx, NCy]);
		roundCreate();
	}
	
} );

document.getElementById("clear").onclick = function(){
	Tmove = [];
	Smove = [];
	Cmove = [];
	Rmove = [];
	rRmove = [];
	triangleIndex = 0;
	squareIndex = 0;
	cubeIndex = 0;
	circleIndex = 0;
};
	document.getElementById("slide").onchange = function(){
		sideNum = document.getElementById("slide").value;
		circleIndex--;
		roundCreate();
	}
    render();

    function triangleCreate(x, y){
		Tmove.push([x, y]);
		triangleIndex++;
    }
    
    function squareCreate(x, y){
		Smove.push([x, y]);
		squareIndex++;
	}

	function cubeCreate(x, y){
		Cmove.push([x, y]);
		cubeIndex++;
	}

	function roundCreate(){
		circleIndex++;
		circleVertex = [];
		circleColors = [];
		var alpha = 2 * Math.PI / sideNum;
		circleVertex.push(0.0, 0.0, 0.0);
		circleColors.push(1.0, 0.0, 0.0, 1.0);
		for(var i=0;i<=sideNum;i++){
			circleVertex.push(0.2 * Math.cos(Math.PI-alpha*i), 0.2 * Math.sin(Math.PI-alpha*i), 0.0);
			circleColors.push(1.0, 0.0, 0.0, 1.0);
		}
		recreate();
	}
}

function btn(){
    var obj = document.getElementsByName("shape");
    for(var i=0; i<obj.length; i++){
		if(obj[i].checked){
			shape = obj[i].value;
		}
	}
}

function render(){
    gl.clear( gl.COLOR_BUFFER_BIT );
    if(shape == "trangle")
    {
        triangleRender();
    }else if(shape == "square"){
        squareRender();
    }else if(shape == "3Dsquare"){
		cubeRender();
	}else if(shape == "round"){
		roundRender();
	}
    window.requestAnimFrame( render );
}

function triangleRender(){
    Stheta[0] -= 0.01;
	if(Stheta[0] < -0.5) Stheta[0] = 0.0;
	gl.uniform3fv(SthetaLoc, Stheta);
	gl.uniform2fv(thetaLoc, [0.0, 0.0]);
	gl.uniform3fv(CthetaLoc, [0.0, 0.0, 0.0]);
	for(var i = 0;i < triangleIndex; i++){
        gl.uniform2fv(moveLoc, Tmove[i]);
		gl.drawArrays(gl.TRIANGLES, 0, 3);
	}	
}

function squareRender(){
	theta += 0.1;
	gl.uniform2fv(thetaLoc, [0.0, theta]);
	gl.uniform3fv(SthetaLoc, [0.0, 0.0, 0.0]);
	gl.uniform3fv(CthetaLoc, [0.0, 0.0, 0.0]);
	for(var i=0;i<squareIndex;i++){
		gl.uniform2fv(moveLoc, Smove[i]);
		gl.drawArrays(gl.TRIANGLE_STRIP, 3, 4);
	}
}
function cubeRender(){
	Ctheta[0] += 0.1;
	gl.uniform3fv(CthetaLoc, Ctheta);
	gl.uniform2fv(thetaLoc, [0.0, 0.0]);
	gl.uniform3fv(SthetaLoc, [0.0, 0.0, 0.0]);
	for( var i=0;i<cubeIndex;i++ ){
		gl.uniform2fv(moveLoc, Cmove[i]);
		gl.drawArrays(gl.TRIANGLES, 7, 36);
	}
}
function roundRender(){
	gl.uniform3fv(CthetaLoc, [0.0, 0.0, 0.0]);
	gl.uniform2fv(thetaLoc, [0.0, 0.0]);
	gl.uniform3fv(SthetaLoc, [0.0, 0.0, 0.0]);
	for(var i=0;i<circleIndex;i++){
		rRmove[i][0] += Math.random()/10 - 0.05;
		rRmove[i][1] += Math.random()/10 - 0.05;		
		if(rRmove[i][0] > 1 || rRmove[i][0] < -1 || rRmove[i][1] > 1 || rRmove[i][1] < -1){
			rRmove[i][0] -= rRmove[i][0]/5;
			rRmove[i][1] -= rRmove[i][1]/5;
		}
		gl.uniform2fv(moveLoc, rRmove[i]);
		gl.drawArrays(gl.TRIANGLE_FAN, 43, sideNum+2);
	}
}

function makeCube() {
    var vertices = [
        glMatrix.vec4.fromValues(-0.1, -0.1, 0.1, 1.0),
        glMatrix.vec4.fromValues(-0.1, 0.1, 0.1, 1.0),
        glMatrix.vec4.fromValues(0.1, 0.1, 0.1, 1.0),
        glMatrix.vec4.fromValues(0.1, -0.1, 0.1, 1.0),
        glMatrix.vec4.fromValues(-0.1, -0.1, -0.1, 1.0),
        glMatrix.vec4.fromValues(-0.1, 0.1, -0.1, 1.0),
        glMatrix.vec4.fromValues(0.1, 0.1, -0.1, 1.0),
        glMatrix.vec4.fromValues(0.1, -0.1, -0.1, 1.0),
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
		ponits.push(vertices[faces[i]][0], vertices[faces[i]][1], vertices[faces[i]][2]);
        colors.push(vertexColors[Math.floor(i / 6)][0], vertexColors[Math.floor(i / 6)][1], vertexColors[Math.floor(i / 6)][2], vertexColors[Math.floor(i / 6)][3]);
    }
}