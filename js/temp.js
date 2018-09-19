let pixels = 28;
let data = new Array(pixels);
for (let i = 0; i < pixels; i++) {
  data[i] = new Array(pixels);
}
for (let i = 0; i < pixels; i++) {
  for (let j = 0; j < pixels; j++) {
    data[i][j] = -1;
  }
}
async function loadtfmodel() {
  const model = await tf.loadModel("./model.json");
  return model;
}
let size = 10;
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
ctx.fillStyle = "#000000";
ctx.fillRect(0, 0, size * pixels, size * pixels);
let state = false;
let init = true;
function getPosition(x, y) {
  return {
    x: Math.floor(x / size) * size,
    y: Math.floor(y / size) * size
  };
}
function indexToValue(dx, dy) {
  let std = 2;
  let gauss = Math.exp(-Math.hypot(dx, dy) / (2 * std ** 2));
  return gauss / 0.5 - 1;
}
function paint(e) {
  let sx = size * pixels;
  let sy = size * pixels;
  let canvas = document.getElementById("canvas");
  let rect = canvas.getBoundingClientRect();
  let ctx = canvas.getContext("2d");
  let coor = getPosition(e.clientX - rect.left, e.clientY - rect.top);
  let x = coor.x;
  let y = coor.y;
  if (state) {
    ctx.beginPath();
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(x, y, size, size);
    x = x / size;
    y = y / size;
    data[y][x] = 1;
    const nx = [0, -1, 0];
    const ny = [-1, 0, 0];
    for (let i = 0; i < 4; i++) {
      let _x = x + nx[i],
        _y = y + ny[i];
      if (_x >= 0 && _x < pixels && (_y >= 0 && _y < pixels)) {
        data[_y][_x] = indexToValue(x - _x, y - _y);
        let col = 255; //Math.round(127.5 * (data[_y][_x] + 1));
        ctx.fillStyle = `rgb(${col}, ${col}, ${col})`;
        ctx.fillRect(_x * size, _y * size, size, size);
      }
    }
  }
  // if(init){
  //     for(let i=0; i < sx; i += size){
  //         for(let j=0; j < sy; j += size){
  //             ctx.strokeRect(i,j,size,size);
  //         }
  //     }
  //     init = false;
  // }
}
async function testing() {
  //   var str;
  //   for (let i = 0; i < 28; i++) {
  //     str = "";
  //     for (let j = 0; j < 28; j++) str += Math.round(data[i][j]) + " ";
  //     console.log(str);
  //   }
  const model = await loadtfmodel();
  const pred = await model.predict(tf.tensor([data]), { batchSize: 1 });
  const num = tf.argMax(pred, -1);
  num.print();
}
function clear() {
  //console.log("yes");
  let canvas = document.getElementById("canvas");
  let ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, size * pixels, size * pixels);
  //init = true;
  for (let i = 0; i < pixels; i++)
    for (let j = 0; j < pixels; j++) data[i][j] = -1;
}
function activate() {
  state = true;
}
function deactivate() {
  state = false;
}
function save() {
  console.log("save");
  let image = new Image();
  let canvas = document.getElementById("canvas");
  let ctx = canvas.getContext("2d");
  alert(ctx.getImageData(0, 0, 280, 280));
  this.href = image.src;
}
