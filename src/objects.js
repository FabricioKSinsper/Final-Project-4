class Shape
{
    name = "Box";
    transform = [0.0, 0.0, 0.0, 0.0];
    radius = [0.0, 0.0, 0.0, 0.0];
    rotation = [0.0, 0.0, 0.0, 0.0];
    op = [0.0, 0.0, 0.0, 0.0];
    color = [0.0, 0.0, 0.0, 0.0];
    animateTransform = [0.0, 0.0, 0.0, 0.0];
    animateRotation = [0.0, 0.0, 0.0, 0.0];
    ignoreData = [0.0, 0.0, 0.0, 0.0];
    ignoreData2 = [0.0, 0.0, 0.0, 0.0];

    constructor(name, transform, radius, rotation, op, color, animateTransform, animateRotation)
    {
        this.name = name;
        this.transform = [transform[0], transform[1], transform[2], 0.0];
        this.radius = [radius[0], radius[1], radius[2], radius[3]];
        this.rotation = [rotation[0], rotation[1], rotation[2], 0.0];
        this.op = [op[0], op[1], op[2], op[3]];
        this.color = [color[0], color[1], color[2], 0.0];
        this.animateTransform = [animateTransform[0], animateTransform[1], animateTransform[2], animateTransform[3]];
        this.animateRotation = [animateRotation[0], animateRotation[1], animateRotation[2], animateRotation[3]];
    }
}

export { Shape };