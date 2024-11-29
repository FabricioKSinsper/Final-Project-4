import { getGPU, getCanvas, getComputeBuffer, getComputeBufferLayout, getModule, getRenderPipeline, RGBToInt, timer, RGBToHex, crateAllScenesList } from "./src/util.js";
import { Shape } from "./src/objects.js";
import { getAvailableScene } from "./src/scenes.js";

// arrays
let spheres = [];
let boxes = [];
let toruses = [];

// variables and constants
const THREAD_COUNT = 16;
const MAX_SPHERES = 20;
const MAX_BOXES = 20;
const MAX_TORUSES = 20;

let sphereTemplate = new Shape("Sphere", [0, 1, 0, 0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0, .2, 0.0, 0.0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]);
let boxTemplate = new Shape("Box", [0, 2, 0], [.5, .5, .5, 0.1], [0, 0, 0], [0, 0.01, 0.0, 0.0], [0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]);
let torusTemplate = new Shape("Torus", [0, 2, 0], [.5, .5, .5, 0.1], [0, 0, 0], [0, 0.01, 0.0, 0.0], [0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]);

let perfCount = 0;
let cameraVelocity = [0.0, 0.0, 0.0];
let cameraRotationVelocity = [0.0, 0.0, 0.0];
let performanceStats = { ms: 0, fps: 0 };

const sizes = { f32: 4, u32: 4, i32: 4, vec2: 8, vec4: 16 };
const uniforms = {
    time: 0, // 0
    rez: 768, // 1
    sphereCount: spheres.length, // 2
    boxCount: boxes.length, // 3
    torusCount: toruses.length, // 4
    maxMarchingSteps: 100, // 5
    camerax: 0, // 6
    cameray: 1.0, // 7
    cameraz: -6.0, // 8
    lookatx: 0, // 9
    lookaty: 1, // 10
    lookatz: 0.0, // 11
    backgroundColor1: 0.0, // 12
    sunx: 4.87, // 13
    suny: 25.34, // 14
    sunz: -28.97, // 15
    sunColor: 1.0, // 16
    showFloor: 1, // 17   
    mandelbulb: 0, // 18
    weirdScene: 0, // 19
    farPlane: 700.0, // 20
    softShadowK: 0.01, // 21
    marchingStep: 1, // 22
    epsilon: 0.01, // 23
    softShadowMin: 0.001, // 24
    softShadowMax: 5.0, // 25
    outlinePostProcess: 0, // 26
    outlineWidth: 0.05, // 27
    outlineColor: 0.0, // 28
    backgroundColor2: 0.0, // 29
    backgroundColor3: 0.0, // 30
}

const uniformsCount = Object.keys(uniforms).length;

// get the GPU and canvas
const { adapter, gpu } = await getGPU();
const { canvas, context, format } = await getCanvas(uniforms.rez);
context.configure({ device: gpu, format: format, alphaMode: "premultiplied", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,});

// create the buffers
const frameBufferSize = sizes.vec4 * uniforms.rez ** 2;
const frameBuffer = await getComputeBuffer(gpu, frameBufferSize, GPUBufferUsage.STORAGE);

const uniformsBufferSize = sizes.f32 * uniformsCount;
const uniformsBuffer = await getComputeBuffer(gpu, uniformsBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE);

const shapesBufferSize = (sizes.vec4 * 9) * (MAX_SPHERES + MAX_BOXES + MAX_TORUSES);
const shapesBuffer = await getComputeBuffer(gpu, shapesBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE);

const shapesInfoBufferSizes = (sizes.vec4) * (MAX_SPHERES + MAX_BOXES + MAX_TORUSES);
const shapesInfoBuffer = await getComputeBuffer(gpu, shapesInfoBufferSizes, GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE);

const frameBuffers = [{ buffer: frameBuffer, type: "storage" }];
const { bindGroupLayout: frameBuffersLayout, bindGroup: frameBuffersBindGroup } = await getComputeBufferLayout(gpu, frameBuffers);

const uniformBuffers = [{ buffer: uniformsBuffer, type: "storage" }];
const { bindGroupLayout: uniformsLayout, bindGroup: uniformsBindGroup } = await getComputeBufferLayout(gpu, uniformBuffers);

const objectBuffers = [
    { buffer: shapesBuffer, type: "storage" }, 
    { buffer: shapesInfoBuffer, type: "storage" }, 
];

// bind group layout
const { bindGroupLayout: objectsLayout, bindGroup: objectsBindGroup } = await getComputeBufferLayout(gpu, objectBuffers);

// memory layout
const layout = gpu.createPipelineLayout({ bindGroupLayouts: [frameBuffersLayout, uniformsLayout, objectsLayout] });

// set the uniforms
const module = await getModule(gpu, ["./src/shaders/utils.wgsl", "./src/shaders/raymarcher.wgsl", "./src/shaders/shapes.wgsl"]);

// get the preprocess kernel
const preprocessKernel = gpu.createComputePipeline({ layout, compute: { module, entryPoint: "preprocess" }, });

// get the render kernel
const renderKernel = gpu.createComputePipeline({ layout, compute: { module, entryPoint: "render" }, });

// get the render pipeline
const { renderPipeline: renderPipeline, bindGroupLayout: renderBindGroupLayout } = await getRenderPipeline(gpu, uniforms.rez, frameBuffer, format);
let availableScenes = await crateAllScenesList();

//!!!! LIL GUI IGNORE THIS
let gui = new lil.GUI();
let performanceFolder = gui.addFolder("Performance");
performanceFolder.add(performanceStats, 'fps').name("fps").listen().disable();
performanceFolder.add(performanceStats, 'ms').name("ms").listen().disable();

let scenesFolder = gui.addFolder("Scene");
const scenesNames = { name: "Main"};

scenesFolder.add(scenesNames, 'name', availableScenes).name("Scene").listen().onChange( function() { getScene(scenesNames.name); });
var cameraVelocityConstant = 5.0;
scenesFolder.add({ CameraVelocity: cameraVelocityConstant }, 'CameraVelocity').name("Camera Velocity").step(0.01).listen().onChange( function() { cameraVelocityConstant = this.object.CameraVelocity; });

scenesFolder.add({ NewScene: () => {
    spheres = [sphereTemplate];
    boxes = [];
    toruses = [];
    
    writeBuffers();
    writeUniforms();
    refreshObjectsGUI(0, true);
}}, 'NewScene');

scenesFolder.add({ ResetScene: () => { getScene(scenesNames.name); }}, 'ResetScene');

scenesFolder.add({ ResetCamera: () => {
    uniforms.camerax = 0;
    uniforms.cameray = 1.0;
    uniforms.cameraz = -6.0;
    uniforms.lookatx = 0;
    uniforms.lookaty = 1;
    uniforms.lookatz = 0.0;
}}, 'ResetCamera');

scenesFolder.add({ SaveImage: () => {
    let link = document.createElement('a');
    link.download = 'image.png';
    link.href = canvas.toDataURL();
    link.click();
}}, 'SaveImage');

scenesFolder.add({ PrintScene: () => {
    let scene = `async function yourSceneName()` + "\n" + "{\n";
    scene += "\treturn {\n";
    scene += "\t\tspheres: [\n";
    for (let i = 0; i < spheres.length; i++)
    {
        let sphere = spheres[i];
        scene += `\t\t\tnew Shape("Sphere", `;
        scene += `[${sphere.transform[0]}, ${sphere.transform[1]}, ${sphere.transform[2]}, ${sphere.transform[3]}], `;
        scene += `[${sphere.radius[0]}, ${sphere.radius[1]}, ${sphere.radius[2]}, ${sphere.radius[3]}], `;
        scene += `[${sphere.rotation[0]}, ${sphere.rotation[1]}, ${sphere.rotation[2]}, ${sphere.rotation[3]}], `;
        scene += `[${sphere.op[0]}, ${sphere.op[1]}, ${sphere.op[2]}, ${sphere.op[3]}], `;
        scene += `[${sphere.color[0]}, ${sphere.color[1]}, ${sphere.color[2]}, ${sphere.color[3]}], `;
        scene += `[${sphere.animateTransform[0]}, ${sphere.animateTransform[1]}, ${sphere.animateTransform[2]}, ${sphere.animateTransform[3]}], `;
        scene += `[${sphere.animateRotation[0]}, ${sphere.animateRotation[1]}, ${sphere.animateRotation[2]}, ${sphere.animateRotation[3]}]`;
        scene += `),\n`;
    }
    scene += "\t\t],\n";

    scene += "\t\tboxes: [\n";
    for (let i = 0; i < boxes.length; i++)
    {
        let box = boxes[i];
        scene += `\t\t\tnew Shape("Box", `;
        scene += `[${box.transform[0]}, ${box.transform[1]}, ${box.transform[2]}, ${box.transform[3]}], `;
        scene += `[${box.radius[0]}, ${box.radius[1]}, ${box.radius[2]}, ${box.radius[3]}], `;
        scene += `[${box.rotation[0]}, ${box.rotation[1]}, ${box.rotation[2]}, ${box.rotation[3]}], `;
        scene += `[${box.op[0]}, ${box.op[1]}, ${box.op[2]}, ${box.op[3]}], `;
        scene += `[${box.color[0]}, ${box.color[1]}, ${box.color[2]}, ${box.color[3]}], `;
        scene += `[${box.animateTransform[0]}, ${box.animateTransform[1]}, ${box.animateTransform[2]}, ${box.animateTransform[3]}], `;
        scene += `[${box.animateRotation[0]}, ${box.animateRotation[1]}, ${box.animateRotation[2]}, ${box.animateRotation[3]}]`;
        scene += `),\n`;
    }
    scene += "\t\t],\n";

    scene += "\t\ttoruses: [\n";
    for (let i = 0; i < toruses.length; i++)
    {
        let torus = toruses[i];
        scene += `\t\t\tnew Shape("Torus", `;
        scene += `[${torus.transform[0]}, ${torus.transform[1]}, ${torus.transform[2]}, ${torus.transform[3]}], `;
        scene += `[${torus.radius[0]}, ${torus.radius[1]}, ${torus.radius[2]}, ${torus.radius[3]}], `;
        scene += `[${torus.rotation[0]}, ${torus.rotation[1]}, ${torus.rotation[2]}, ${torus.rotation[3]}], `;
        scene += `[${torus.op[0]}, ${torus.op[1]}, ${torus.op[2]}, ${torus.op[3]}], `;
        scene += `[${torus.color[0]}, ${torus.color[1]}, ${torus.color[2]}, ${torus.color[3]}], `;
        scene += `[${torus.animateTransform[0]}, ${torus.animateTransform[1]}, ${torus.animateTransform[2]}, ${torus.animateTransform[3]}], `;
        scene += `[${torus.animateRotation[0]}, ${torus.animateRotation[1]}, ${torus.animateRotation[2]}, ${torus.animateRotation[3]}]`;
        scene += `),\n`;
    }

    scene += "\t\t],\n";
    scene += `\t\tbackgroundColor: [${backgroundColor1.object.r}, ${backgroundColor1.object.g}, ${backgroundColor1.object.b}],\n`;
    scene += `\t\tmaxMarchingSteps: ${uniforms.maxMarchingSteps},\n`;
    scene += `\t\tshowFloor: ${uniforms.showFloor},\n`;
    scene += `\t\tmandelbulb: ${uniforms.mandelbulb},\n`;
    scene += `\t\tweirdScene: ${uniforms.weirdScene},\n`;
    scene += `\t\tfarPlane: ${uniforms.farPlane},\n`;
    scene += `\t\tsoftShadowK: ${uniforms.softShadowK},\n`;
    scene += `\t\tmarchingStep: ${uniforms.marchingStep},\n`;
    scene += `\t\toutlinePostProcess: ${uniforms.outlinePostProcess},\n`;
    scene += `\t\toutlineWidth: ${uniforms.outlineWidth},\n`;
    scene += `\t\tsunColor: [${sunColor.object.r}, ${sunColor.object.g}, ${sunColor.object.b}],\n`;
    scene += `\t\tbackgroundColor2: [${backgroundColor2.object.r}, ${backgroundColor2.object.g}, ${backgroundColor2.object.b}],\n`;
    scene += `\t\tbackgroundColor3: [${backgroundColor3.object.r}, ${backgroundColor3.object.g}, ${backgroundColor3.object.b}],\n`;
    scene += "\t};\n";
    scene += "}\n";
    
    console.log(scene);
    navigator.clipboard.writeText(scene);
}}, 'PrintScene');

let uniformsFolder = gui.addFolder("Uniforms");
uniformsFolder.add(uniforms, 'maxMarchingSteps').name("Max Marching Steps").step(1).listen();
uniformsFolder.add(uniforms, 'farPlane').name("Far Plane").step(1).listen();
uniformsFolder.add(uniforms, 'epsilon').name("Epsilon").step(0.0001).listen();
uniformsFolder.add(uniforms, 'marchingStep').name("Marching Step").step(0.01).listen();
uniformsFolder.add({ ShowFloor: true }, 'ShowFloor').name("Show Floor").listen().onChange( function() { uniforms.showFloor = this.object.ShowFloor ? 1 : 0; });
uniformsFolder.add({ Mandelbulb: false }, 'Mandelbulb').name("Mandelbulb").listen().onChange( function() { uniforms.mandelbulb = this.object.Mandelbulb ? 1 : 0; });
uniformsFolder.add({ WeirdScene: false }, 'WeirdScene').name("Weird Scene").listen().onChange( function() { uniforms.weirdScene = this.object.WeirdScene ? 1 : 0; });

let shadowsFolder = gui.addFolder("Shadows");
shadowsFolder.add(uniforms, 'softShadowK').name("Soft Shadow K").step(0.01).listen();
shadowsFolder.add(uniforms, 'softShadowMin').name("Soft Shadow Min").step(0.001).listen();
shadowsFolder.add(uniforms, 'softShadowMax').name("Soft Shadow Max").step(0.01).listen();

let outlineFolder = gui.addFolder("Outline");
outlineFolder.add({ OutlinePostProcess: false }, 'OutlinePostProcess').name("Outline Post Process").listen().onChange( function() { uniforms.outlinePostProcess = this.object.OutlinePostProcess ? 1 : 0; });
outlineFolder.add(uniforms, 'outlineWidth').name("Outline Width").step(0.01).listen();

const outlineColor = {
    outlineColor: '#FFFFFF',
    object: { r: 1, g: 1, b: 1 },
};

outlineFolder.addColor( outlineColor, 'outlineColor' ).onChange( function() {
    outlineColor.object.r = parseInt(outlineColor.outlineColor.substring(1, 3), 16) / 255;
    outlineColor.object.g = parseInt(outlineColor.outlineColor.substring(3, 5), 16) / 255;
    outlineColor.object.b = parseInt(outlineColor.outlineColor.substring(5, 7), 16) / 255;
}).listen();

let sunFolder = gui.addFolder("Sun");
sunFolder.add(uniforms, 'sunx').name("Sun X").step(0.01).listen();
sunFolder.add(uniforms, 'suny').name("Sun Y").step(0.01).listen();
sunFolder.add(uniforms, 'sunz').name("Sun Z").step(0.01).listen();

const sunColor = {
    sunColor: '#FFFFFF',
    object: { r: 1, g: 1, b: 1 },
};

sunFolder.addColor( sunColor, 'sunColor' ).onChange( function() {
    sunColor.object.r = parseInt(sunColor.sunColor.substring(1, 3), 16) / 255;
    sunColor.object.g = parseInt(sunColor.sunColor.substring(3, 5), 16) / 255;
    sunColor.object.b = parseInt(sunColor.sunColor.substring(5, 7), 16) / 255;
}).listen();

let background = gui.addFolder("Background Color");
const backgroundColor1 = { BGC1: '#FFFFFF', object: { r: 1, g: 1, b: 1 },};
const backgroundColor2 = { BGC2: '#FFFFFF', object: { r: 1, g: 1, b: 1 },};
const backgroundColor3 = { BGC3: '#FFFFFF', object: { r: 1, g: 1, b: 1 },};

function CreateBackgroundColorGUI(object, name)
{
    background.addColor( object, name ).onChange( function() {
        object.object.r = parseInt(object[name].substring(1, 3), 16) / 255;
        object.object.g = parseInt(object[name].substring(3, 5), 16) / 255;
        object.object.b = parseInt(object[name].substring(5, 7), 16) / 255;
    }
    ).listen();
}

CreateBackgroundColorGUI(backgroundColor1, 'BGC1');
CreateBackgroundColorGUI(backgroundColor2, 'BGC2');
CreateBackgroundColorGUI(backgroundColor3, 'BGC3');

// background.addColor( backgroundColor1, 'BGC1' ).onChange( function() {
//     backgroundColor1.object.r = parseInt(backgroundColor1.BGC1.substring(1, 3), 16) / 255;
//     backgroundColor1.object.g = parseInt(backgroundColor1.BGC1.substring(3, 5), 16) / 255;
//     backgroundColor1.object.b = parseInt(backgroundColor1.BGC1.substring(5, 7), 16) / 255;
// }).listen();


// background.addColor( backgroundColor2, 'BGC2' ).onChange( function() {
//     backgroundColor2.object.r = parseInt(backgroundColor2.BGC2.substring(1, 3), 16) / 255;
//     backgroundColor2.object.g = parseInt(backgroundColor2.BGC2.substring(3, 5), 16) / 255;
//     backgroundColor2.object.b = parseInt(backgroundColor2.BGC2.substring(5, 7), 16) / 255;
// }).listen();


// background.addColor( backgroundColor3, 'BGC3' ).onChange( function() {
//     backgroundColor3.object.r = parseInt(backgroundColor3.BGC3.substring(1, 3), 16) / 255;
//     backgroundColor3.object.g = parseInt(backgroundColor3.BGC3.substring(3, 5), 16) / 255;
//     backgroundColor3.object.b = parseInt(backgroundColor3.BGC3.substring(5, 7), 16) / 255;
// }).listen();

let objectsFolder = gui.addFolder("Objects");

function refreshObjectGUI(template, objectList, parentFolder, folder, startIndex, hasButtons = true)
{
    let objectFolder = folder;
    if (!objectFolder)
    {
        objectFolder = parentFolder.addFolder(template.name + "(s)");
        
        if (hasButtons)
        {
            objectFolder.add({ AddObject: () => {
                let copyTemplate = JSON.parse(JSON.stringify(template));
                objectList.push(copyTemplate);
                refreshObjectGUI(template, objectList, parentFolder, objectFolder, objectList.length - 1);
                writeBuffers();
            }}, 'AddObject');
        }

        objectFolder.close();
    }

    let variables = Object.keys(template);

    for (let i = startIndex; i < objectList.length; i++)
    {
        let folder = objectFolder.addFolder(`${template.name} ${i + 1}`);
        for (let j = 1; j < variables.length; j++)
        {
            if (variables[j].includes("ignoreData"))
            {
                continue;
            }

            let folderVariable = folder.addFolder(variables[j]);
            let lengthOfVariable = 0;
            if (typeof template[variables[j]] === 'object')
            {
                lengthOfVariable = template[variables[j]].length;
            }
            switch (variables[j])
            {
                case "color":
                    let colorObj = {
                        Hex: RGBToHex(objectList[i].color[0] * 255, objectList[i].color[1] * 255, objectList[i].color[2] * 255),
                    };

                    folderVariable.addColor(colorObj, 'Hex').onChange( function() {
                        objectList[i].color[0] = parseInt(colorObj.Hex.substring(1, 3), 16) / 255;
                        objectList[i].color[1] = parseInt(colorObj.Hex.substring(3, 5), 16) / 255;
                        objectList[i].color[2] = parseInt(colorObj.Hex.substring(5, 7), 16) / 255;
                    });

                    folderVariable.close();
                    break;
                default:
                    if (lengthOfVariable == 0)
                    {
                        folderVariable.add(objectList[i], variables[j]).step(0.01).listen();
                        folderVariable.close();
                        break;
                    }

                    for (let k = 0; k < lengthOfVariable; k++)
                    {
                        folderVariable.add(objectList[i][variables[j]], k.toString()).name(k == 0 ? "X" : k == 1 ? "Y" : k == 2 ? "Z" : "W").step(0.01).listen();
                    }

                    folderVariable.close();
                    break;
            }
        }

        folder.close();
    }
}

function refreshObjectsGUI(startIndex = 0, rebuild = false)
{
    if (rebuild)
    {
        objectsFolder.destroy();
        objectsFolder = gui.addFolder("Objects");
    }

    refreshObjectGUI(sphereTemplate, spheres, objectsFolder, null, startIndex);
    refreshObjectGUI(boxTemplate, boxes, objectsFolder, null, startIndex);
    refreshObjectGUI(torusTemplate, toruses, objectsFolder, null, startIndex);
}

gui.onFinishChange( event => {});
gui.onChange( event => { writeBuffers();});
//!!!! LIL GUI IGNORE THIS

document.addEventListener("keydown", checkUserInput);
document.addEventListener("keyup", checkUserInput);

window.addEventListener("keydown", function(e) {
    if(["Space","ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].indexOf(e.code) > -1) {
        e.preventDefault();
    }
}, false);

let rotQuat = quat.create();

function checkUserInput(event)
{
    let isKeyUp = event.type == "keyup";
    var velocity = cameraVelocityConstant;

    var setVel = (axis, event, value, originalVal) => {
        if (event.key == axis)
        {
            return isKeyUp ? 0.0 : value;
        }

        return originalVal;
    }

    cameraVelocity[0] = setVel("a", event, -velocity, cameraVelocity[0]);
    cameraVelocity[0] = setVel("d", event, velocity, cameraVelocity[0]);
    cameraVelocity[1] = setVel("q", event, -velocity, cameraVelocity[1]);
    cameraVelocity[1] = setVel("e", event, velocity, cameraVelocity[1]);
    cameraVelocity[2] = setVel("w", event, -velocity, cameraVelocity[2]);
    cameraVelocity[2] = setVel("s", event, velocity, cameraVelocity[2]);
    cameraRotationVelocity[2] = setVel("z", event, -velocity, cameraRotationVelocity[2]);
    cameraRotationVelocity[2] = setVel("x", event, velocity, cameraRotationVelocity[2]);
    cameraRotationVelocity[0] = setVel("ArrowLeft", event, -velocity, cameraRotationVelocity[0]);
    cameraRotationVelocity[0] = setVel("ArrowRight", event, velocity, cameraRotationVelocity[0]);
    cameraRotationVelocity[1] = setVel("ArrowDown", event, -velocity, cameraRotationVelocity[1]);
    cameraRotationVelocity[1] = setVel("ArrowUp", event, velocity, cameraRotationVelocity[1]);
}

function generateBackgroundAndSunColor(rgb1, rgb2, rgb3, sun)
{
    backgroundColor1.object = { r: rgb1[0], g: rgb1[1], b: rgb1[2] };
    backgroundColor1.BGC1 = RGBToHex(rgb1[0] * 255, rgb1[1] * 255, rgb1[2] * 255);

    backgroundColor2.object = { r: rgb2[0], g: rgb2[1], b: rgb2[2] };
    backgroundColor2.BGC2 = RGBToHex(rgb2[0] * 255, rgb2[1] * 255, rgb2[2] * 255);

    backgroundColor3.object = { r: rgb3[0], g: rgb3[1], b: rgb3[2] };
    backgroundColor3.BGC3 = RGBToHex(rgb3[0] * 255, rgb3[1] * 255, rgb3[2] * 255);

    sunColor.object = { r: sun[0], g: sun[1], b: sun[2] };
    sunColor.sunColor = RGBToHex(sun[0] * 255, sun[1] * 255, sun[2] * 255);
}

async function getScene(index)
{
    let backgroundColor, maxMarchingSteps, 
        showFloor, mandelbulb, weirdScene, 
        farPlane, softShadowK, marchingStep, 
        outlinePostProcess, outlineWidth, sunColor, backgroundColor2, backgroundColor3;

    ({ 
        spheres, boxes, toruses, 
        backgroundColor, maxMarchingSteps, 
        showFloor, mandelbulb, weirdScene, 
        farPlane, softShadowK, marchingStep, 
        outlinePostProcess, outlineWidth, sunColor, backgroundColor2, backgroundColor3
    } = await getAvailableScene(index, availableScenes));

    uniforms.maxMarchingSteps = maxMarchingSteps;
    uniforms.showFloor = showFloor;
    uniforms.mandelbulb = mandelbulb;
    uniforms.weirdScene = weirdScene;
    uniforms.farPlane = farPlane;
    uniforms.softShadowK = softShadowK;
    uniforms.marchingStep = marchingStep;
    uniforms.outlinePostProcess = outlinePostProcess;
    uniforms.outlineWidth = outlineWidth;

    generateBackgroundAndSunColor(backgroundColor, backgroundColor2, backgroundColor3, sunColor);
    writeBuffers();
    writeUniforms();
    refreshObjectsGUI(0, true);
}

function setup()
{
    getScene(0);
}

function writeUniforms()
{
    // uniforms
    uniforms.time = performance.now() / 1000;
    uniforms.sunColor = RGBToInt(sunColor.object.r * 255, sunColor.object.g * 255, sunColor.object.b * 255);
    uniforms.outlineColor = RGBToInt(outlineColor.object.r * 255, outlineColor.object.g * 255, outlineColor.object.b * 255);
    uniforms.backgroundColor1 = RGBToInt(backgroundColor1.object.r * 255, backgroundColor1.object.g * 255, backgroundColor1.object.b * 255);
    uniforms.backgroundColor2 = RGBToInt(backgroundColor2.object.r * 255, backgroundColor2.object.g * 255, backgroundColor2.object.b * 255);
    uniforms.backgroundColor3 = RGBToInt(backgroundColor3.object.r * 255, backgroundColor3.object.g * 255, backgroundColor3.object.b * 255);
    uniforms.sphereCount = spheres.length;
    uniforms.boxCount = boxes.length;
    uniforms.torusCount = toruses.length;
    
    var uniformData = new Float32Array(uniformsBufferSize / sizes.f32);
    var offset = 0;

    for (let key in uniforms)
    {
        uniformData[offset++] = uniforms[key];
    }

    gpu.queue.writeBuffer(uniformsBuffer, 0, uniformData);
}

function writeBuffer(buffer, size, objectList)
{
    if (objectList.length == 0)
    {
        return;
    }
    
    var objectData = new Float32Array(size / sizes.f32);
    var offset = 0;

    var variables = Object.keys(objectList[0]);
    for (let i = 0; i < objectList.length; i++)
    {
        for (let j = 0; j < variables.length; j++)
        {
            if (typeof objectList[i][variables[j]] === 'string')
            {
                continue;
            }

            let lengthOfVariable = 0;
            if (typeof objectList[0][variables[j]] === 'object')
            {
                lengthOfVariable = objectList[0][variables[j]].length;
            }
            
            if (lengthOfVariable == 0)
            {
                objectData[offset++] = objectList[i][variables[j]];
                continue;
            }

            for (let k = 0; k < lengthOfVariable; k++)
            {
                objectData[offset++] = objectList[i][variables[j]][k];
            }
        }
    }

    gpu.queue.writeBuffer(buffer, 0, objectData);
}

// set the uniforms
function writeBuffers()
{
    // shapes
    var shapes = [];
    shapes = shapes.concat(spheres);
    shapes = shapes.concat(boxes);
    shapes = shapes.concat(toruses);

    writeBuffer(shapesBuffer, shapesBufferSize, shapes);

    var shapesInfo = [];
    for (let i = 0; i < shapes.length; i++)
    {
        switch (shapes[i].name)
        {
            case "Sphere":
                shapesInfo.push([0, i, shapes[i].op[0], 0]);
                break;
            case "Box":
                shapesInfo.push([1, i, shapes[i].op[0], 0]);
                break;
            case "Torus":
                shapesInfo.push([2, i, shapes[i].op[0], 0]);
                break;
        }
    }

    // sort the shapes by the operation
    shapesInfo.sort((a, b) => a[2] - b[2]);
    writeBuffer(shapesInfoBuffer, shapesInfoBufferSizes, shapesInfo);
}

// render framebuffer to quad
function renderToScreen(encoder)
{
    var renderPass = encoder.beginRenderPass({
        colorAttachments: [
            {
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: "clear",
                storeOp: "store",
            },
        ],
    });

    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroupLayout);
    renderPass.draw(6, 1, 0, 0);
    renderPass.end();
}

// dispatch the compute render pass
function dispatchComputeRenderPass(pass)
{
    pass.setBindGroup(0, frameBuffersBindGroup);
    pass.setBindGroup(1, uniformsBindGroup);
    pass.setBindGroup(2, objectsBindGroup);

    var workgroupPre = math.ceil((spheres.length + boxes.length + toruses.length) / THREAD_COUNT);
    pass.setPipeline(preprocessKernel);
    pass.dispatchWorkgroups(workgroupPre < 1 ? 1 : workgroupPre, 1, 1);

    pass.setPipeline(renderKernel);
    pass.dispatchWorkgroups(uniforms.rez / THREAD_COUNT, uniforms.rez / THREAD_COUNT, 1);

    pass.end();
}

function moveCamera(deltaTime)
{
    var rotatedCameraVelocity = vec3.create();
    vec3.transformQuat(rotatedCameraVelocity, cameraVelocity, rotQuat);

    var rotatedCameraRotationVelocity = vec3.create();
    vec3.transformQuat(rotatedCameraRotationVelocity, cameraRotationVelocity, rotQuat);

    uniforms.camerax += rotatedCameraVelocity[0] * deltaTime;
    uniforms.cameray += rotatedCameraVelocity[1] * deltaTime;
    uniforms.cameraz += rotatedCameraVelocity[2] * deltaTime;

    uniforms.lookatx += rotatedCameraRotationVelocity[0] * deltaTime;
    uniforms.lookaty += rotatedCameraRotationVelocity[1] * deltaTime;
    uniforms.lookatz += rotatedCameraRotationVelocity[2] * deltaTime;

    var rotMatrix = mat4.create();
    mat4.targetTo(rotMatrix, [uniforms.camerax, uniforms.cameray, uniforms.cameraz], [uniforms.lookatx, uniforms.lookaty, uniforms.lookatz], [0, 1, 0]);
    mat4.getRotation(rotQuat, rotMatrix);
}

// update and render
async function update()
{
    let startms = performance.now();

    // begin the compute pass
    const encoder = gpu.createCommandEncoder();
    const pass = encoder.beginComputePass();

    // set the uniforms
    writeUniforms();

    // render the scene
    dispatchComputeRenderPass(pass);
    renderToScreen(encoder);

    // end the compute pass
    gpu.queue.submit([encoder.finish()]);
    await gpu.queue.onSubmittedWorkDone();

    // get fps
    perfCount ++;
    let elapsedms = (performance.now() - startms);
    
    if (perfCount == 60)
    {
        performanceStats.ms = elapsedms.toFixed(2);
        performanceStats.fps = (1 / elapsedms * 1000).toFixed(0);
        perfCount = 0;
    }

    moveCamera(elapsedms / 1000);
    window.requestAnimationFrame(update, 0);
};

setup();
update();