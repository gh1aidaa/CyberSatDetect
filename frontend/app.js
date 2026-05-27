document.addEventListener('DOMContentLoaded', () => {
  let renderer, scene, camera, sphereBg, nucleus, stars, controls, clock;
  const container = document.getElementById("canvas_container");
  const noise = new SimplexNoise();
  let delta = 0;
  let blobScale = 3;

  init();
  animate();

  function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.01, 1000);
    camera.position.set(0, 0, 230);

    const directionalLight = new THREE.DirectionalLight("#fff", 1.2);
    directionalLight.position.set(0, 50, -20);
    scene.add(directionalLight);

    const ambientLight = new THREE.AmbientLight("#ffffff", 1);
    scene.add(ambientLight);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    clock = new THREE.Clock();

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.autoRotate = true;
    controls.autoRotateSpeed = 3;
    controls.maxDistance = 350;
    controls.minDistance = 150;
    controls.enablePan = false;

    const loader = new THREE.TextureLoader();
    const textureSphereBg = loader.load('https://i.ibb.co/HC0vxMw/sky2.jpg');
    const texturenucleus = loader.load('https://i.ibb.co/hcN2qXk/star-nc8wkw.jpg');
    const textureStar = loader.load("https://i.ibb.co/ZKsdYSz/p1-g3zb2a.png");

    const sphereGeo = new THREE.SphereGeometry(150, 40, 40);
    const sphereMat = new THREE.MeshBasicMaterial({ side: THREE.BackSide, map: textureSphereBg });
    sphereBg = new THREE.Mesh(sphereGeo, sphereMat);
    scene.add(sphereBg);

    const nucleusGeo = new THREE.IcosahedronGeometry(30, 10);
    const nucleusMat = new THREE.MeshPhongMaterial({ map: texturenucleus });
    nucleus = new THREE.Mesh(nucleusGeo, nucleusMat);
    scene.add(nucleus);

    const starsGeo = new THREE.BufferGeometry();
    const starCount = 100;
    const positions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      const x = (Math.random() - 0.5) * 300;
      const y = (Math.random() - 0.5) * 300;
      const z = (Math.random() - 0.5) * 300;
      positions.set([x, y, z], i * 3);
    }
    starsGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const starsMat = new THREE.PointsMaterial({
      size: 6,
      map: textureStar,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0.8
    });
    stars = new THREE.Points(starsGeo, starsMat);
    scene.add(stars);

    window.addEventListener("resize", onWindowResize);

    const fsEnter = document.querySelector(".fullscr");
    let fullscreen = false;
    fsEnter.addEventListener("click", e => {
      e.preventDefault();
      if (!fullscreen) {
        fullscreen = true;
        document.documentElement.requestFullscreen();
        fsEnter.innerHTML = "Exit Fullscreen";
      } else {
        fullscreen = false;
        document.exitFullscreen();
        fsEnter.innerHTML = "Go Fullscreen";
      }
    });
  }

  function animate() {
    requestAnimationFrame(animate);

    const time = Date.now() * 0.0004;
    const position = nucleus.geometry.attributes.position;
    const vertex = new THREE.Vector3();

    for (let i = 0; i < position.count; i++) {
      vertex.fromBufferAttribute(position, i);
      vertex.normalize();
      const distance = 30 + noise.noise3D(vertex.x + time, vertex.y + time, vertex.z + time) * blobScale;
      vertex.multiplyScalar(distance);
      position.setXYZ(i, vertex.x, vertex.y, vertex.z);
    }
    position.needsUpdate = true;

    nucleus.rotation.y += 0.002;
    sphereBg.rotation.y += 0.002;
    controls.update();
    renderer.render(scene, camera);
  }

  function onWindowResize() {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }
});

