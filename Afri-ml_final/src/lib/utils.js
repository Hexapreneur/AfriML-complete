import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

function cn(...inputs) {
  return twMerge(clsx(inputs))
}

function coordinates(connections2) {
  const paths = document.getElementById("paths");
  let padding = 1;
  let bezierWeight = 0.5;
  let oldPaths = paths.children;

  for (let a = oldPaths.length - 1; a >= 0; a--) {
    paths.removeChild(oldPaths[a]);
  }

  let x1, y1, x4, y4, dx, x2, x3, path, start, end;
  end = document.querySelector(".end");

  for (let a = 0; a < connections2.length; a++) {
    start = connections2[a];
    // console.log("connections2", start);

    let startRect = start.getBoundingClientRect();
    let endRect = end.getBoundingClientRect();
    x1 = startRect.left + start.offsetWidth;

    //get start span top position then go to its center and remove top banner height and add scroll from top
    y1 = startRect.top + start.offsetHeight / 2 - 320 + scrollY;
    x4 = endRect.left;

    //get end span top position then go to its center and remove top banner height and add scroll from top
    y4 = endRect.top + end.offsetHeight / 2 - padding - 320 + scrollY;
    dx = Math.abs(x4 - x1) * bezierWeight;

    if (x4 < x1) {
      x2 = x1 - dx;
      x3 = x4 + dx;
    } else {
      x2 = x1 + dx;
      x3 = x4 - dx;
    }

    let data = `M${x1} ${y1} C ${x2} ${y1} ${x3} ${y4} ${x4} ${y4}`;

    path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", data);
    path.setAttribute("class", "path");
    paths.appendChild(path);
  }
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");

  //preview card is start
  start = document.querySelector(".preview");

  let startRect = start.getBoundingClientRect();
  //training card is end
  let endRect = end.getBoundingClientRect();

  //line should start from training card x1, y1
  // add scrolling
  x1 = endRect.left + end.offsetWidth;
  y1 = endRect.top + end.offsetHeight / 2 - 320 + scrollY;

  //line should end at preview card x1, y1
  x4 = startRect.left;
  y4 = startRect.top + start.offsetHeight / 2 - 320 + scrollY;
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x4);
  line.setAttribute("y2", 448);
  line.setAttribute("class", "path");
  paths.appendChild(line);
}

const connectElements = (setConnections2) => {
  setConnections2([]);
  const paths = document.getElementById("paths");
  const mvSequenceDiv = document.querySelector(".mv-sequence");
  if (mvSequenceDiv) {
    const mvSequenceRect = mvSequenceDiv.getBoundingClientRect();
    paths.setAttribute("height", mvSequenceRect.height);
    paths.setAttribute("width", mvSequenceRect.width);
  }
  let spans = document.getElementsByClassName("mv-command1");
  const start = [];
  for (let b = 0; b < spans.length; b++) {
    start.push(spans[b]);
  }
  setConnections2(start);
}

export {
  cn,
  coordinates,
  connectElements,
}