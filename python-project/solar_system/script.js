document.addEventListener('DOMContentLoaded', () => {
    const solarSystemContainer = document.getElementById('solar-system-container');
    const sun = document.getElementById('sun');

    // Center of the solar system (sun's position)
    const centerX = sun.offsetLeft + sun.offsetWidth / 2;
    const centerY = sun.offsetTop + sun.offsetHeight / 2;

    const planetsData = [
        { id: 'mercury', element: document.getElementById('mercury'), radius: 70, period: 5, angle: Math.random() * 2 * Math.PI, size: 10 }, // Radius in pixels, period in seconds
        { id: 'venus', element: document.getElementById('venus'), radius: 100, period: 8, angle: Math.random() * 2 * Math.PI, size: 15 },
        { id: 'earth', element: document.getElementById('earth'), radius: 140, period: 12, angle: Math.random() * 2 * Math.PI, size: 20 },
        { id: 'mars', element: document.getElementById('mars'), radius: 180, period: 18, angle: Math.random() * 2 * Math.PI, size: 12 },
        { id: 'jupiter', element: document.getElementById('jupiter'), radius: 250, period: 30, angle: Math.random() * 2 * Math.PI, size: 40 },
        { id: 'saturn', element: document.getElementById('saturn'), radius: 320, period: 45, angle: Math.random() * 2 * Math.PI, size: 35 },
        { id: 'uranus', element: document.getElementById('uranus'), radius: 380, period: 60, angle: Math.random() * 2 * Math.PI, size: 25 },
        { id: 'neptune', element: document.getElementById('neptune'), radius: 430, period: 75, angle: Math.random() * 2 * Math.PI, size: 22 },
    ];

    function positionPlanets() {
        planetsData.forEach(planet => {
            if (planet.element) {
                // Initial position calculation
                const x = centerX + planet.radius * Math.cos(planet.angle) - planet.size / 2;
                const y = centerY + planet.radius * Math.sin(planet.angle) - planet.size / 2;
                planet.element.style.left = `${x}px`;
                planet.element.style.top = `${y}px`;
            }
        });
    }

    let lastTimestamp = 0;
    function animate(timestamp) {
        const deltaTime = (timestamp - lastTimestamp) / 1000; // Time in seconds
        lastTimestamp = timestamp;

        planetsData.forEach(planet => {
            if (planet.element) {
                // Calculate new angle based on orbital period
                // Speed = 2 * PI / period
                planet.angle += (2 * Math.PI / planet.period) * deltaTime;
                if (planet.angle > 2 * Math.PI) {
                    planet.angle -= 2 * Math.PI;
                }

                // Calculate new position
                const x = centerX + planet.radius * Math.cos(planet.angle) - planet.element.offsetWidth / 2;
                const y = centerY + planet.radius * Math.sin(planet.angle) - planet.element.offsetHeight / 2;

                // Update DOM
                planet.element.style.left = `${x}px`;
                planet.element.style.top = `${y}px`;
            }
        });

        requestAnimationFrame(animate);
    }

    // Initial setup
    positionPlanets(); // Position planets initially based on their angles
    requestAnimationFrame(animate); // Start the animation loop
});
