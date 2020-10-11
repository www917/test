"use strict";

const { vec3 } = glMatrix;

var canvas;
var gl;
var shape="2D";
var points = [];
var colors = [];
var numTimesToSubdivide = 4;
var theta;
var radius=1.0;
var rotate;

function init(){	
	canvas = document.getElementById( "gl-canvas" );
	gl = WebGLUtils.setupWebGL( canvas );
	if( !gl ){
		alert( "WebGL isn't available" );
	}
	if(shape=="2D")
	{
		var vertices = [
			-1, -1,  0,
			 0,  1,  0,
			 1, -1,  0
		];
		var u = vec3.fromValues( vertices[0], vertices[1], vertices[2] );
		var v = vec3.fromValues( vertices[3], vertices[4], vertices[5] );
		var w = vec3.fromValues( vertices[6], vertices[7], vertices[8] );
		changeCase();
		divideTriangle2D( u, v, w, numTimesToSubdivide );
	
		// configure webgl
		gl.viewport( 0, 0, canvas.width, canvas.height );
		gl.clearColor( 1.0, 1.0, 1.0, 1.0 );
	
		// load shaders and initialise attribute buffers
		var program = initShaders( gl, "vertex-shader2D", "fragment-shader2D" );
		gl.useProgram( program );
	
		// load data into gpu
		var vertexBuffer = gl.createBuffer();
		gl.bindBuffer( gl.ARRAY_BUFFER, vertexBuffer );
		gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( points ), gl.STATIC_DRAW );
	
		// associate out shader variables with data buffer
		var vPosition = gl.getAttribLocation( program, "vPosition" );
		gl.vertexAttribPointer( vPosition, 3, gl.FLOAT, false, 0, 0 );
		gl.enableVertexAttribArray( vPosition );
	
		renderTriangles2D();
	}else if(shape == "3D"){
		var vertices = [
			0.0000, 0.0000, -1.0000,
			0.0000, 0.9428, 0.3333,
			-0.8165, -0.4714, 0.3333,
			0.8165, -0.4714, 0.3333
		];
	
		var t = vec3.fromValues( vertices[0], vertices[1], vertices[2] );
		var u = vec3.fromValues( vertices[3], vertices[4], vertices[5] );
		var v = vec3.fromValues( vertices[6], vertices[7], vertices[8] );
		var w = vec3.fromValues( vertices[9], vertices[10], vertices[11] );
		
		changeCase();
		divideTetra3D(t, u, v, w, numTimesToSubdivide);
	
		// configure webgl
		gl.viewport(0, 0, canvas.width, canvas.height);
		gl.clearColor(1.0, 1.0, 1.0, 1.0);
	
		// enable hidden-surface removal
		gl.enable(gl.DEPTH_TEST);
	
		// load shaders and initialize attribute buffers
		var program = initShaders(gl, "vertex-shader3D", "fragment-shader3D");
		gl.useProgram(program);

		var vBuffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points), gl.STATIC_DRAW);
	
		var vPosition = gl.getAttribLocation(program, "vPosition");
		gl.vertexAttribPointer(vPosition, 3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(vPosition);
	
		var cBuffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, cBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
	
		var vColor = gl.getAttribLocation(program, "vColor");
		gl.vertexAttribPointer(vColor, 3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(vColor);
	
		render();
	}
	else{
		canvas = document.getElementById( "gl-canvas" );
		gl = WebGLUtils.setupWebGL( canvas );
		if( !gl ){
			alert( "WebGL isn't available" );
		}
		if(rotate=="plane"||rotate=="dis"){
			changeangle();
			var vertices = [
			radius * Math.cos(90 * Math.PI / 180.0), radius * Math.sin(90 * Math.PI / 180.0),  0,
			radius * Math.cos(210 * Math.PI / 180.0), radius * Math.sin(210 * Math.PI / 180.0),  0,
			radius * Math.cos(-30 * Math.PI / 180.0), radius * Math.sin(-30 * Math.PI / 180.0),  0
		];
		}
		else{
			var vertices = [
				-1, -1,  0,
				 0,  1,  0,
				 1, -1,  0
			];
		}
		var u = vec3.fromValues( vertices[0], vertices[1], vertices[2] );
		var v = vec3.fromValues( vertices[3], vertices[4], vertices[5] );
		var w = vec3.fromValues( vertices[6], vertices[7], vertices[8] );
		
		divideTrianglexk( u, v, w, numTimesToSubdivide );
	
		// configure webgl
		gl.viewport( 0, 0, canvas.width, canvas.height );
		gl.clearColor( 1.0, 1.0, 1.0, 1.0 );
	
		// load shaders and initialise attribute buffers
		var program = initShaders( gl, "vertex-shader2D", "fragment-shader2D" );
		gl.useProgram( program );
	
		// load data into gpu
		var vertexBuffer = gl.createBuffer();
		gl.bindBuffer( gl.ARRAY_BUFFER, vertexBuffer );
		gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( points ), gl.STATIC_DRAW );
	
		// associate out shader variables with data buffer
		var vPosition = gl.getAttribLocation( program, "vPosition" );
		gl.vertexAttribPointer( vPosition, 3, gl.FLOAT, false, 0, 0 );
		gl.enableVertexAttribArray( vPosition );
	
		renderTrianglesxk();
	}
};

function changeangle(){
	points=[];
	theta = document.getElementById("switch").value;
	document.getElementById("demo1").innerHTML = "角度: " + theta;
	console.log(theta);
}

function changeCase(){
	points=[];
	numTimesToSubdivide = document.getElementById("slide").value;
	document.getElementById("demo").innerHTML = "层数: " + numTimesToSubdivide;
	console.log(numTimesToSubdivide);
}

function btn(){
	points=[];
	var obj = document.getElementsByName("shape");
	for(var i=0; i<obj.length; i++){
		if(obj[i].checked){
			shape = obj[i].value;
		}
	}
}

function btn1(){
	points=[];
	var obj = document.getElementsByName("rotate");
	for(var i=0; i<obj.length; i++){
		if(obj[i].checked){
			rotate = obj[i].value;
		}
	}
}

function triangle2D( a, b, c ){
	//var k;
	points.push( a[0], a[1], a[2] );
	points.push( b[0], b[1], b[2] );
	points.push( c[0], c[1], c[2] );
}

function divideTriangle2D( a, b, c, count ){
	// check for end of recursion
	if( count == 0 ){
		triangle2D( a, b, c );
	}else{
		var ab = vec3.create();
		vec3.lerp( ab, a, b, 0.5 );
		var bc = vec3.create();
		vec3.lerp( bc, b, c, 0.5 );
		var ca = vec3.create();
		vec3.lerp( ca, c, a, 0.5 );

		// three new triangles
		divideTriangle2D( a, ab, ca, count-1 );
		divideTriangle2D( b, bc, ab, count-1 );
		divideTriangle2D( c, ca, bc, count-1 );
	}
}

function renderTriangles2D(){
	gl.clear( gl.COLOR_BUFFER_BIT );
	gl.drawArrays( gl.TRIANGLES, 0, points.length/3 );
}

function triangle3D(a, b, c, color) {
    // add colors and vertices for one triangle
    var baseColor = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0
    ];

    for (var k = 0; k < 3; k++) {
        colors.push(baseColor[color * 3 + k]);
    }
    for (var k = 0; k < 3; k++)
        points.push(a[k]);

    for (var k = 0; k < 3; k++) {
        colors.push(baseColor[color * 3 + k]);
    }
    for (var k = 0; k < 3; k++)
        points.push(b[k]);

    for (var k = 0; k < 3; k++) {
        colors.push(baseColor[color * 3 + k]);
    }
    for (var k = 0; k < 3; k++)
        points.push(c[k]);
}

function tetra3D(a, b, c, d) {
    triangle3D(a, c, b, 0);
    triangle3D(a, c, d, 1);
    triangle3D(a, b, d, 2);
    triangle3D(b, c, d, 3);
}

function divideTetra3D(a, b, c, d, count) {
    // check for end of recursion
    if (count == 0) {
        tetra3D(a, b, c, d);
    } else {
        var ab = vec3.create();
        glMatrix.vec3.lerp(ab, a, b, 0.5);
        var ac = vec3.create();
        glMatrix.vec3.lerp(ac, a, c, 0.5);
        var ad = vec3.create();
        glMatrix.vec3.lerp(ad, a, d, 0.5);
        var bc = vec3.create();
        glMatrix.vec3.lerp(bc, b, c, 0.5);
        var bd = vec3.create();
        glMatrix.vec3.lerp(bd, b, d, 0.5);
        var cd = vec3.create();
        glMatrix.vec3.lerp(cd, c, d, 0.5);

        --count;

        divideTetra3D(a, ab, ac, ad, count);
        divideTetra3D(ab, b, bc, bd, count);
        divideTetra3D(ac, bc, c, cd, count);
        divideTetra3D(ad, bd, cd, d, count);
    }

}

function render() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, points.length / 3);
}

function trianglexk( a, b, c ){
	//var k;
	points.push( a[0], a[1], a[2] );
	points.push( b[0], b[1], b[2] );

	points.push( b[0], b[1], b[2] );
	points.push( c[0], c[1], c[2] );

	points.push( c[0], c[1], c[2] );
	points.push( a[0], a[1], a[2] );
}

function divideTrianglexk( a, b, c, count ){
	// check for end of recursion
	if( count == 0 ){
		if(rotate=="plane"||rotate=="dis")
		{
			tessellaTriangle( a, b, c )
		}
		else{
			trianglexk( a, b, c );
		}
	}else{
		var ab = vec3.create();
		vec3.lerp( ab, a, b, 0.5 );
		var bc = vec3.create();
		vec3.lerp( bc, b, c, 0.5 );
		var ca = vec3.create();
		vec3.lerp( ca, c, a, 0.5 );

		// three new triangles
		divideTrianglexk( a, ab, ca, count-1 );
		divideTrianglexk( b, bc, ab, count-1 );
		divideTrianglexk( c, ca, bc, count-1 );
		divideTrianglexk( ab, bc, ca, count-1 );
	}
}

function renderTrianglesxk(){
	gl.clear( gl.COLOR_BUFFER_BIT );
	gl.drawArrays( gl.LINES, 0, points.length/3 );
}

function tessellaTriangle( a, b, c ){
    //var k;
    var zerovec3 = vec3.create();
    vec3.zero( zerovec3 );
    var radian = theta * Math.PI / 180.0;
    
    var a_new = vec3.create();
    var b_new = vec3.create();
    var c_new = vec3.create();

    if( rotate=="plane" ){
        vec3.rotateZ( a_new, a, zerovec3, radian );
        vec3.rotateZ( b_new, b, zerovec3, radian );
        vec3.rotateZ( c_new, c, zerovec3, radian );
        
        points.push( a_new[0], a_new[1], a_new[2] );
        points.push( b_new[0], b_new[1], b_new[2] );
        points.push( b_new[0], b_new[1], b_new[2] );
        points.push( c_new[0], c_new[1], c_new[2] );
        points.push( c_new[0], c_new[1], c_new[2] );
        points.push( a_new[0], a_new[1], a_new[2] );
    }else{
        var d_a = Math.sqrt( a[0] * a[0] + a[1] * a[1] );
        var d_b = Math.sqrt( b[0] * b[0] + b[1] * b[1] );
        var d_c = Math.sqrt( c[0] * c[0] + c[1] * c[1] );

        vec3.set( a_new, a[0] * Math.cos(d_a * radian) - a[1] * Math.sin( d_a * radian ), 
            a[0] * Math.sin( d_a * radian ) + a[1] * Math.cos( d_a * radian ), 0 );
        vec3.set(b_new, b[0] * Math.cos(d_b * radian) - b[1] * Math.sin(d_b * radian),
            b[0] * Math.sin(d_b * radian) + b[1] * Math.cos(d_b * radian), 0);
        vec3.set(c_new, c[0] * Math.cos(d_c * radian) - c[1] * Math.sin(d_c * radian),
            c[0] * Math.sin(d_c * radian) + c[1] * Math.cos(d_c * radian), 0);
        
        points.push(a_new[0], a_new[1], a_new[2]);
        points.push(b_new[0], b_new[1], b_new[2]);
        points.push(b_new[0], b_new[1], b_new[2]);
        points.push(c_new[0], c_new[1], c_new[2]);
        points.push(c_new[0], c_new[1], c_new[2]);
        points.push(a_new[0], a_new[1], a_new[2]);
    
    }
}