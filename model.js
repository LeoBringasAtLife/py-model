    function pad4(n) {
      return String(n).padStart(4, '0');
    }

    function createImagePath(split, digit, index) {
      return `data/${split}/${digit}/${digit}_${pad4(index)}.png`;
    }

    function renderGallery() {
      const split = document.getElementById('split').value;
      const count = Math.max(1, Math.min(200, Number(document.getElementById('count').value) || 12));
      const gallery = document.getElementById('gallery');
      gallery.innerHTML = '';

      for (let digit = 0; digit <= 9; digit++) {
        const block = document.createElement('section');
        block.className = 'digit-block';

        const title = document.createElement('h2');
        title.className = 'digit-title';
        title.textContent = `Digit ${digit}`;
        block.appendChild(title);

        const grid = document.createElement('div');
        grid.className = 'grid';

        for (let i = 0; i < count; i++) {
          const tile = document.createElement('div');
          tile.className = 'tile';

          const img = document.createElement('img');
          img.src = createImagePath(split, digit, i);
          img.alt = `Digit ${digit} sample ${i}`;
          img.loading = 'lazy';

          tile.appendChild(img);
          grid.appendChild(tile);
        }

        block.appendChild(grid);
        gallery.appendChild(block);
      }
    }

    document.getElementById('render').addEventListener('click', renderGallery);
    renderGallery();