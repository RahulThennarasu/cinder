import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const FeatureImportance3D = ({ featureImportance }) => {
  const mountRef = useRef(null);
  const [hoveredFeature, setHoveredFeature] = useState(null);
  
  useEffect(() => {
    if (!featureImportance || !featureImportance.feature_names) {
      return;
    }
    
    // Scene setup
    const width = mountRef.current.clientWidth;
    const height = 300;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f5f7);
    
    // Camera setup
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;
    camera.position.y = 1;
    
    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    mountRef.current.appendChild(renderer.domElement);
    
    // Add controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Grid helper
    const gridHelper = new THREE.GridHelper(10, 10, 0x888888, 0x444444);
    scene.add(gridHelper);
    
    // Prepare feature data
    const features = featureImportance.feature_names.slice(0, 10);
    const importanceValues = featureImportance.importance_values.slice(0, 10);
    
    // Create feature bars
    const bars = [];
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    
    features.forEach((feature, index) => {
      const importance = importanceValues[index];
      const scaledHeight = importance * 4; // Scale for better visualization
      const color = new THREE.Color().setHSL(0.05, 0.8, 0.5);
      
      // Bar geometry
      const geometry = new THREE.BoxGeometry(0.4, scaledHeight, 0.4);
      
      // Bar material with custom shader for hover effect
      const material = new THREE.MeshStandardMaterial({
        color: color,
        roughness: 0.5,
        metalness: 0.2,
      });
      
      // Create mesh and position it
      const bar = new THREE.Mesh(geometry, material);
      bar.position.x = index - 4.5; // Center the bars
      bar.position.y = scaledHeight / 2; // Position from the bottom
      
      // Store the feature info with the bar
      bar.userData = { feature, importance };
      
      // Add to scene and tracking array
      scene.add(bar);
      bars.push(bar);
      
      // Add text label
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      canvas.width = 64;
      canvas.height = 32;
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.font = '12px Arial';
      context.fillStyle = '#000000';
      context.textAlign = 'center';
      context.fillText(feature, canvas.width / 2, canvas.height / 2);
      
      const texture = new THREE.CanvasTexture(canvas);
      const labelMaterial = new THREE.SpriteMaterial({ map: texture });
      const label = new THREE.Sprite(labelMaterial);
      label.position.set(index - 4.5, -0.3, 0);
      label.scale.set(1, 0.5, 1);
      scene.add(label);
    });
    
    // Handle hover interaction
    const handleMouseMove = (event) => {
      // Calculate mouse position in normalized device coordinates
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / height) * 2 + 1;
      
      // Update the picking ray with the camera and mouse position
      raycaster.setFromCamera(mouse, camera);
      
      // Calculate objects intersecting the picking ray
      const intersects = raycaster.intersectObjects(bars);
      
      // Reset all bars to original color
      bars.forEach(bar => {
        bar.material.color.setHSL(0.05, 0.8, 0.5);
        bar.material.emissive.set(0x000000);
      });
      
      // If we found an intersection, highlight it
      if (intersects.length > 0) {
        const bar = intersects[0].object;
        bar.material.color.setHSL(0.05, 1, 0.7);
        bar.material.emissive.set(0x331100);
        setHoveredFeature(bar.userData);
      } else {
        setHoveredFeature(null);
      }
    };
    
    // Add event listener
    renderer.domElement.addEventListener('mousemove', handleMouseMove);
    
    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    
    // Handle resize
    const handleResize = () => {
      const width = mountRef.current.clientWidth;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('mousemove', handleMouseMove);
      mountRef.current.removeChild(renderer.domElement);
      
      // Dispose of resources
      scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach(material => material.dispose());
          } else {
            object.material.dispose();
          }
        }
      });
      
      renderer.dispose();
    };
  }, [featureImportance]);
  
  return (
    <div className="feature-importance-3d">
      <h3 className="chart-title">Feature Importance</h3>
      <div className="chart-description">
        <p>Examine which features have the greatest impact on predictions for this class.</p>
        <p className="chart-tip">Drag to rotate | Scroll to zoom | Hover over bars for details</p>
      </div>
      
      <div 
        ref={mountRef} 
        className="feature-importance-container"
        style={{ width: '100%', height: '300px', position: 'relative' }}
      ></div>
      
      {hoveredFeature && (
        <div className="feature-tooltip">
          <div className="tooltip-title">{hoveredFeature.feature}</div>
          <div className="tooltip-importance">
            Importance: <span>{hoveredFeature.importance.toFixed(3)}</span>
          </div>
          <div className="tooltip-description">
            Higher values indicate greater influence on model predictions
          </div>
        </div>
      )}
      
      <style jsx>{`
        .feature-importance-3d {
          padding: 1rem;
          background-color: #ffffff;
          border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .chart-title {
          margin-top: 0;
          margin-bottom: 0.5rem;
          font-size: 1.2rem;
          font-weight: 600;
        }
        
        .chart-description {
          margin-bottom: 1rem;
          font-size: 0.9rem;
          color: #666;
        }
        
        .chart-tip {
          font-size: 0.8rem;
          color: #888;
          font-style: italic;
          margin-top: 0.5rem;
        }
        
        .feature-tooltip {
          position: absolute;
          bottom: 10px;
          left: 10px;
          background-color: rgba(255, 255, 255, 0.9);
          border-left: 3px solid #e74c32;
          padding: 8px 12px;
          border-radius: 4px;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
          pointer-events: none;
          z-index: 100;
          max-width: 250px;
        }
        
        .tooltip-title {
          font-weight: 600;
          margin-bottom: 4px;
        }
        
        .tooltip-importance {
          font-size: 0.9rem;
          margin-bottom: 4px;
        }
        
        .tooltip-importance span {
          font-weight: 600;
          color: #e74c32;
        }
        
        .tooltip-description {
          font-size: 0.8rem;
          color: #666;
        }
      `}</style>
    </div>
  );
};

export default FeatureImportance3D;