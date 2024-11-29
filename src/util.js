async function getGPU()
{
    var adapter = await navigator.gpu.requestAdapter();
    var gpu = await adapter.requestDevice();

    return {adapter, gpu};
}

async function getCanvas(rez)
{
    var canvas = document.createElement("canvas");
    var context = canvas.getContext("webgpu");
    var format = "bgra8unorm";
    canvas.style = `padding-left: 0; padding-right: 0; margin-left: auto; margin-right: auto;`;

    canvas.width = canvas.height = rez;
    document.body.appendChild(canvas);
    
    return {canvas, context, format};
}

async function getComputeBuffer(gpu, size, usage)
{
    var buffer = gpu.createBuffer({
        size: size,
        usage: usage,
    });

    return buffer;
}

async function getComputeBufferLayout(gpu, buffers)
{
    var entriesLayout = [];
    var entriesGroup = [];

    for (let i = 0; i < buffers.length; i++)
    {
        entriesLayout.push({
            visibility: GPUShaderStage.COMPUTE,
            binding: i,
            buffer: { type: buffers[i].type },
        });

        entriesGroup.push({
            binding: i,
            resource: { buffer: buffers[i].buffer },
        });
    }

    var bindGroupLayout = gpu.createBindGroupLayout({
        entries: entriesLayout,
    });

    var bindGroup = gpu.createBindGroup({
        layout: bindGroupLayout,
        entries: entriesGroup,
    });

    return { bindGroupLayout, bindGroup };
}

async function load(file)
{
    var response = await fetch(file);
    if (!response.ok)
    {
        throw new Error(`Error loading: ${file}`);
    }

    return await response.text();
}

async function getModule(gpu, paths)
{
    var code = "";
    for (let path of paths)
    {
        code += await load(path) + "\n";
    }

    var module = gpu.createShaderModule({ code });
    var info = await module.getCompilationInfo();

    if (info.messages.length > 0)
    {
        for (let message of info.messages)
        {
            console.warn(`${message.message} at ${file} line ${message.lineNum}`);
        }

        throw new Error(`Could not compile ${file}`);
    }

    return module;
}

async function getRenderPipeline(gpu, rez, buffer, format)
{
    var quadShader = 
    `
        @group(0) @binding(0)  
        var<storage, read_write> pixels : array<vec4f>;

        struct VertexOutput {
        @builtin(position) Position : vec4f,
            @location(0) fragUV : vec2f,
        }
        
        @vertex
        fn vert(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
        
        const pos = array(
            vec2( 1.0,  1.0),
            vec2( 1.0, -1.0),
            vec2(-1.0, -1.0),
            vec2( 1.0,  1.0),
            vec2(-1.0, -1.0),
            vec2(-1.0,  1.0),
        );
        
        const uv = array(
            vec2(1.0, 0.0),
            vec2(1.0, 1.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
            vec2(0.0, 1.0),
            vec2(0.0, 0.0),
        );

        var output : VertexOutput;
        output.Position = vec4(pos[VertexIndex], 0.0, 1.0);
        output.fragUV = uv[VertexIndex];
        return output;
        }
        
        @fragment
        fn frag(@location(0) fragUV : vec2f) -> @location(0) vec4f {
        var color = vec4(0, 0, 0, 1.0);
        color += pixels[i32((fragUV.x * ${rez}) + floor(fragUV.y * ${rez}) * ${rez})];
        return color;
        }
    `;

    var quadShaderModule = gpu.createShaderModule({ code: quadShader });
    var renderPipeline = gpu.createRenderPipeline({
        layout: "auto",
        vertex: {
          module: quadShaderModule,
          entryPoint: "vert",
        },
        fragment: {
          module: quadShaderModule,
          entryPoint: "frag",
          targets: [
            {
              format: format,
            },
          ],
        },
        primitive: {
          topology: "triangle-list",
        },
      });

    var bindGroupLayout = gpu.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: buffer,
              offset: 0,
              size: rez * rez * 16,
            },
          },
        ],
      });

    return { renderPipeline, bindGroupLayout };
}

function RGBToInt(r, g, b)
{
    return (r << 16) | (g << 8) | b;
}

function RGBToHex(r, g, b)
{
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

function IntToRGB(int)
{
    let r = ((int >> 16) & 0xFF) / 255.0;
    let g = ((int >> 8) & 0xFF) / 255.0;
    let b = (int & 0xFF) / 255.0;

    return [r, g, b];
}

async function readObj(file)
{
    let vertices = [];
    let triangles = [];
    let fileContents = await load(file);
    let lines = fileContents.split("\n");

    for (let line of lines)
    {
        let parts = line.split(" ");
        if (parts[0] == "v")
        {
            vertices.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]), 0.0]);
        }

        if (parts[0] == "f")
        {
            let face = [];
            for (let i = 1; i < parts.length; i++)
            {
                face.push(parseInt(parseInt(parts[i]) - 1));
            }
            
            triangles.push(face);
        }
    }

    return { vertices, triangles };
}

async function loadMesh(file)
{
    let { vertices: verticesMesh, triangles: trianglesMesh } = await readObj(file);
    let triangles = [];

    for (let i = 0; i < trianglesMesh.length; i ++)
    {
        let A = verticesMesh[trianglesMesh[i][0]];
        let B = verticesMesh[trianglesMesh[i][1]];
        let C = verticesMesh[trianglesMesh[i][2]];
        triangles.push(new Triangle(A, B, C));
    
        if (trianglesMesh[i].length == 4)
        {
            A = verticesMesh[trianglesMesh[i][0]];
            B = verticesMesh[trianglesMesh[i][2]];
            C = verticesMesh[trianglesMesh[i][3]];
    
            triangles.push(new Triangle(A, B, C));
        }
    }

    return { verticesMesh, trianglesMesh, triangles };
}

function getObjBoundingBox(vertices)
{
    let min = [Infinity, Infinity, Infinity];
    let max = [-Infinity, -Infinity, -Infinity];

    for (let vertex of vertices)
    {
        for (let i = 0; i < 3; i++)
        {
            min[i] = Math.min(min[i], vertex[i]);
            max[i] = Math.max(max[i], vertex[i]);
        }
    }

    return { min, max };
}

function timer(ms, callback)
{
    return new Promise(resolve => setTimeout(() => { resolve(callback()); }, ms));
}

async function getFunctionsInFile(filepath) {

    let fileContent = await load(filepath);

    const functionRegex = /function\s+([\w$]+)\s*\(|([\w$]+)\s*=\s*\(.*?\)\s*=>|([\w$]+)\s*:\s*function\s*\(/g;

    let match;
    const functionNames = [];

    while ((match = functionRegex.exec(fileContent)) !== null) {
        const functionName = match[1] || match[2] || match[3];
        if (functionName) {
            functionNames.push(functionName);
        }
    }

    if (functionNames.length === 0)
    {
        console.logWarning("No functions found in file: " + filepath);
    }

    return functionNames;
}

async function crateAllScenesList()
{
    let functions = await getFunctionsInFile('./src/scenes.js');

    let sceneList = {};
    let funcCount = 0;
    for (let i = 0; i < functions.length; i++)
    {
        if (functions[i] == "getAvailableScene")
        {
            continue;
        }

        sceneList[functions[i]] = funcCount;
        funcCount++;
    }

    return sceneList;
}

export { getGPU, getCanvas, getComputeBuffer, getComputeBufferLayout, load, getModule, getRenderPipeline, readObj, timer, getObjBoundingBox, RGBToInt, IntToRGB, RGBToHex, loadMesh, crateAllScenesList };