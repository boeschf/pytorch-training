# Images are tensors

Adapted from `slides/src/2.1-featurisation/section-slides.md`.

<img src="/legacy/cnn/image_rgb_as_data.png" style="height: 330px; margin: auto" alt="RGB image represented by three numeric channels" />

```text
single grayscale image  [height, width]
RGB image               [channels, height, width]
model batch             [batch, channels, height, width]
```

---

# A convolution detects local structure

A small kernel is reused at every spatial position. The same learned weights can detect a pattern wherever it occurs.

<div class="grid grid-cols-2 gap-4">
  <img src="/legacy/cnn/sobel_h.png" style="height: 260px; margin: auto" alt="Horizontal Sobel response" />
  <img src="/legacy/cnn/sobel_v.png" style="height: 260px; margin: auto" alt="Vertical Sobel response" />
</div>

The Sobel examples are fixed filters. A CNN learns its filters from the loss.

---

# Shape arithmetic first

For stride 1 and padding 1, a $3\times3$ convolution preserves height and width.

```text
[batch, 1, 8, 8]
  Conv2d(1, 4, kernel_size=3, padding=1)
[batch, 4, 8, 8]
  MaxPool2d(2)
[batch, 4, 4, 4]
```

Channels describe learned feature maps; pooling reduces spatial resolution.
