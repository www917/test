"use strict";

var gl;
var points;

window.onload = function init(){
	var canvas = document.getElementById( "triangle-canvas" );
	gl = WebGLUtils.setupWebGL( canvas );
	//g1 = getWebGLContext(canvas);
	if( !gl ){
		alert( "WebGL isn't available" );
	}
	// Three Vertices
	var verticesColors = [
		-1.0, -1.0,
		 0.0,  1.0,
		 1.0, -1.0,
		 1.0,  0.0,  0.0,
		 0.0,  1.0,  0.0,
		 0.0,  0.0,  1.0,
	];
	var FSIZE = verticesColors.BYTES_PER_ELEMENT;
	// 配置WebGL
	gl.viewport( 0, 0, canvas.width, canvas.height );
	gl.clearColor( 1.0, 1.0, 1.0, 1.0 );

	// 加载着色器和初始化属性缓冲
	var program = initShaders( gl, "vertex-shader", "fragment-shader" );
	gl.useProgram( program );

	// 将数据加载到GPU中
	var bufferId = gl.createBuffer();
	gl.bindBuffer( gl.ARRAY_BUFFER, bufferId );
	gl.bufferData( gl.ARRAY_BUFFER, new Float32Array(verticesColors), gl.STATIC_DRAW );

	// 关联外部着色器变量与数据缓冲区
	var a_Position = gl.getAttribLocation( program, "a_Position" );
	gl.vertexAttribPointer( a_Position, 2, gl.FLOAT, false, FSIZE*5, 0 );
	gl.enableVertexAttribArray( a_Position );
	var a_Color = gl.getAttribLocation(program, "a_Color");
	gl.vertexAttribPointer(a_Color, 3, gl.FLOAT, false, FSIZE*5, FSIZE * 2);
	gl.enableVertexAttribArray(a_Color);
	
	render();
}

function render(){
	gl.clear( gl.COLOR_BUFFER_BIT );
	//gl.drawArrays( gl.TRIANGLE_FAN, 0, 5 );
	gl.drawArrays( gl.TRIANGLES, 0, 3 );
	//gl.drawArrays( gl.TRIANGLE_FANS, 3, 6 );
}


