"use strict";

var canvas;
var gl;

var theta = 0.0;
var thetaLoc;
var direction = 1;
var speed = 50;
var flag=1;

function changeState(){
	flag = flag*-1;
}

function changeDir(){
	direction *= -1;
}

function initRotSquare(){
	canvas = document.getElementById( "rot-canvas" );
	gl = WebGLUtils.setupWebGL( canvas, "experimental-webgl" );
	if( !gl ){
		alert( "WebGL isn't available" );
	}

	gl.viewport( 0, 0, canvas.width, canvas.height );
	gl.clearColor( 1.0, 1.0, 1.0, 1.0 );

	var program = initShaders( gl, "rot-v-shader", "rot-f-shader" );
	gl.useProgram( program );

	var vertices = [
		0.02,0.5,
		0.05,0.45,

		-0.01,0.3,
		-0.05,0.2,

		0.02,0,
		0.05,-0.08,

		-0.01,-0.2,
		-0.05,-0.3,

		0.02,-0.5,
		0.02,0.5,
		-0.01,0.5,	
		-0.01,-0.5,
		0.02,0.5,
		-0.01,0.5,
		-0.01,-0.5,
		0.02,-0.5,

		0.03,0.4,
		0.06,0.5,
		0.08,0.13,

		-0.045,0.15,
		-0.055,0.25,
		-0.14,0.0,

		0.03,-0.1,
		0.06,-0.06,
		0.08,-0.32,

		-0.045,-0.2,
		-0.055,-0.35,
		-0.14,-0.5,

		0,0.5,
		0,-0.5,
		0.01,0.5,
		0.01,-0.5

	];

	var bufferId = gl.createBuffer();
	gl.bindBuffer( gl.ARRAY_BUFFER, bufferId );
	gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( vertices ), gl.STATIC_DRAW );

	var vPosition = gl.getAttribLocation( program, "vPosition" );
	gl.vertexAttribPointer( vPosition, 2, gl.FLOAT, false, 0, 0 );
	gl.enableVertexAttribArray( vPosition );

	thetaLoc = gl.getUniformLocation( program, "theta" );

	document.getElementById( "speedcon" ).onchange = function( event ){
		speed = 100 - event.target.value;
	}

	renderSquare();
}

function renderSquare(){
	gl.clear( gl.COLOR_BUFFER_BIT );
	
	// set uniform values
	if(flag==-1){
		theta += direction * 0.1;
	
		gl.uniform1f( thetaLoc, theta );
	}
	gl.drawArrays( gl.LINES,0,16 );
	gl.drawArrays( gl.TRIANGLES,16,28 );
	gl.drawArrays( gl.LINES,28,30 );
	
	// update and render
	setTimeout( function(){ requestAnimFrame( renderSquare ); }, speed );
}