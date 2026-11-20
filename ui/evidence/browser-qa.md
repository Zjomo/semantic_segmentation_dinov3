# CoreSegment UI Browser QA

## Contract

- Product surface: local map review for stored chip scans, segmentation layers, skeletons, cutting coordinates, and motion logs.
- Primary task: inspect the chip-wide map, compare its overlays, scrub or play the stored path, and export review data.
- Audience and tone: an optics or lab operator who reviews scan results; compact, technical, and suited to repeated inspection.
- Primary viewports: 1440x900 desktop and 390x844 mobile. Also checked at 340x740.
- Existing product evidence: `scan_output/ToupView/test4/` and `scan_output/Map/` contain the maps and saved scan artifacts displayed here.
- Boundary: the page previews saved data; it does not run inference or send commands to a microscope or motion controller.

## Rendered Views

- 1440x900: Chrome DevTools viewport screenshot was visually reviewed. The left rail, chip image with its tool strip, path preview, and right-side sample and motion inspector fit within the viewport. Measured document width was 1440 px, with a 903 px main track and 266 px inspector.
- 390x844: Chrome DevTools viewport screenshot was visually reviewed. Navigation collapses, the map and timeline stay legible, and the inspector flows below. Measured document width was 390 px with no horizontal overflow.
- 340x740: toolbar controls wrap cleanly; measured document width was 340 px with no horizontal overflow.
- Screenshot saving through the browser tool was denied because its file boundary did not include this workspace. These observations come from the browser's rendered screenshots and live DOM checks; this file records the browser-test evidence rather than claiming to be a screenshot.

## Interaction Results

- The default `ToupView / test4` sample loaded the 3568x1532 map, 1516 cutting points, 19356 skeleton pixels, and 13 motion records. The `Map` sample loaded the 8228x7596 map, 24759 route points, 119291 skeleton pixels, and 937 motion records.
- Switching segmentation and skeleton layers updated their image overlays. The path slider and step controls updated the current coordinate and motion row. Zoom controls changed the displayed map size.
- A delayed data fetch showed the loading placeholder. Rejected fetch requests showed the error state and kept the image-import recovery action available; reloading the sample recovered successfully.
- Importing a browser-created image `File` exercised the same `FileReader` flow as the file picker. The image resolution updated, missing path and motion data remained marked as unassociated, the unrelated segmentation and skeleton layers were disabled, and path playback reported that no coordinates were available. Zoom and image export remained available for the custom map.
- The selected sample restored successfully after the custom-image and error-state checks.

## Accessibility And Responsive Checks

- The accessibility tree exposed a named page heading, navigation, regions, dataset selector, range control, checkboxes, descriptive buttons, and image alternatives.
- Tab navigation focused the `导入底图` button and showed its focus ring.
- All visible interactive controls had an accessible name in the browser DOM check.
- At 1440x900, 390x844, and 340x740, document width matched viewport width. Long sample paths remain in the scrollable motion list.

## Content States

- Loading: passed; delayed requests kept the loading placeholder visible until the sample finished loading.
- Empty: passed; an imported map without route coordinates had zero motion records, a clear empty-record message, and a no-path preview response.
- Error: passed; rejected local-data requests showed a recovery action and succeeded after returning to the sample.
- Partial: passed; an imported map rendered without an associated mask, skeleton, path, or motion log, and unavailable layers were disabled.
- Success: passed; both stored scan samples populated their map and metadata views.
- Permission: not applicable; the local inspector has no account, role, or project-access gate.

## Known Limits

- Segmentation inference, model training, hardware capture, and microscope movement are not connected to this UI.
- Imported images require matching path and layer files to review overlays; this page marks those layers as unassociated rather than inventing them.
- No automated frontend test runner is configured in the repository. Verification used the rendered page, accessibility snapshot, live DOM measurements, and browser interaction checks.
