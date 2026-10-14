// This optional artifact builder uses the desktop app's bundled presentation runtime.
// It is separate from the Python scientific pipeline and does not retrain models.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';
import { FontLibrary } from 'skia-canvas';

const repo = path.resolve(process.argv[2] || '.');
const build = path.resolve(process.argv[3] || path.join(repo, 'outputs/presentation-build-20261014'));
const finalName = process.argv[4] || '深度学习综合实践汇报-v1.pptx';
const skill = process.env.PRESENTATIONS_SKILL_DIR;
const python = process.env.RUNTIME_PYTHON;
if (!skill || !python || !process.env.RUNTIME_NODE_MODULES) {
  throw new Error('Provide PRESENTATIONS_SKILL_DIR, RUNTIME_PYTHON and RUNTIME_NODE_MODULES from the bundled runtime.');
}
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(
  path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const spec = JSON.parse(await fs.readFile(path.join(repo, 'docs/presentation-content.json'), 'utf8'));
if (spec.slides.length !== 24 || spec.slides.slice(0, 20).reduce((sum, s) => sum+s.seconds, 0) !== 500) {
  throw new Error('Unexpected fixed slide count or main timing.');
}
const font = spec.font_family;
if (!FontLibrary.families.includes(font)) throw new Error(`Required installed font unavailable: ${font}`);
await fs.mkdir(build, {recursive:true});
const final = path.join(repo, 'deliverables/presentation', finalName);
await fs.mkdir(path.dirname(final), {recursive:true});
await fs.access(final).then(() => { throw new Error('Use a new final filename; do not overwrite.'); }, () => {});

const hashes = {};
for (const source of spec.source_files) {
  hashes[source] = createHash('sha256').update(await fs.readFile(path.join(repo,source))).digest('hex');
}
for (const item of spec.slides) {
  for (const source of item.sources) {
    if (!source.startsWith('https://')) await fs.access(path.join(repo,source.split('#')[0]));
  }
}
const p = Presentation.create({slideSize:{width:1600,height:900}});
const palette = {ink:'#15263C', blue:'#184D88', muted:'#526174', line:'#D7DEE7', red:'#A33E2D'};
const tableSlides = [], chartSlides = [];
function text(slide, value, x, y, w, h, size=32, color=palette.ink, bold=false, name='text') {
  const shape=slide.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},
    fill:'none',line:{fill:'none',width:0}});
  shape.text=value;
  shape.text.style={typeface:font,fontSize:size,color,bold,wrap:'square',autoFit:'none',
    verticalAlignment:'top',insets:0};
  return shape;
}
function sourceLink(source) {
  if (source.startsWith('https://')) return source;
  return `https://github.com/zty666547/deep-learning/blob/${spec.evidence_revision}/${source}`;
}
function columns(item, count) {
  if (item.id==='S07') return [510,330,400,260];
  if (item.id==='S11') return [260,660,230,350];
  if (item.id==='S09') return [300,520,680];
  if (item.id==='S18') return [370,840,290];
  if (item.id==='S19') return [280,640,580];
  if (item.id==='S08') return [560,520,420];
  if (item.id==='A04') return [640,220,640];
  return count===2 ? [360,1140] : count===3 ? [400,550,550] : count===4 ? [390,370,370,370] : [440,265,265,265,265];
}
function table(slide,item,number,y=235,height=445) {
  const values=[item.columns,...item.rows];
  const t=slide.tables.add({rows:values.length,columns:item.columns.length,
    left:50,top:y,width:1500,height,columnWidths:columns(item,item.columns.length),values});
  t.styleOptions={headerRow:true,bandedRows:false};
  t.borders.assign({fill:palette.line,width:1,style:'solid'});
  for (let row=0;row<values.length;row++) {
    t.rows[row].height=height/values.length;
    for (let column=0;column<item.columns.length;column++) {
      const c=t.getCell(row,column);
      c.fill=row===0 ? palette.blue : '#FFFFFF';
      c.text.style={typeface:font,fontSize:row===0?32:31,bold:row===0,
        color:row===0?'#FFFFFF':palette.ink,verticalAlignment:'middle',insets:10};
    }
  }
  tableSlides.push(number);
  return t;
}
async function image(slide, source, position, crop) {
  const bytes=await fs.readFile(path.join(repo,source));
  hashes[source]=createHash('sha256').update(bytes).digest('hex');
  slide.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),
    contentType:'image/png',alt:source,fit:crop?'cover':'contain',position,...(crop?{crop}:{})});
}
for (let index=0;index<spec.slides.length;index++) {
  const item=spec.slides[index], number=index+1;
  const slide=p.slides.add();
  slide.background.fill=item.kind==='cover'?palette.ink:'#FFFFFF';
  if (item.kind==='cover') {
    text(slide,item.title,100,235,1420,225,72,'#FFFFFF',true,'title');
    text(slide,item.subtitle,105,525,1100,70,36,'#DCE7F5',false,'subtitle');
  } else {
    text(slide,item.title,50,50,1500,110,58,palette.ink,true,'title');
    text(slide,String(number).padStart(2,'0'),1470,850,80,32,23,palette.muted,false,'page-number');
    if (item.kind==='table' || item.kind==='confusion') {
      const h=item.id==='S18'?520:item.rows.length===5?470:445;
      table(slide,item,number,210,h);
      if (item.body) text(slide,item.body,55,705,1470,85,31,
        ['S17','S19'].includes(item.id)?palette.red:palette.muted,false,'interpretation');
    } else if (item.kind==='chart') {
      const chart=slide.charts.add('bar',{
        position:{left:50,top:215,width:1010,height:500},
        categories:item.categories,series:[{name:item.series_name,values:item.values,fill:palette.blue}],
        titlePlacement:'none',hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
        chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',
        xAxis:{textStyle:{fontSize:29,typeface:font,fill:palette.ink},line:{fill:palette.line,width:1}},
        yAxis:{min:0,max:item.axis_max,numberFormatCode:item.axis_max===1?'0.0':'0',
          textStyle:{fontSize:28,typeface:font,fill:palette.muted},
          majorGridlines:{fill:palette.line,width:1},title:{text:item.series_name,textStyle:{fontSize:29,typeface:font}}},
        dataLabels:{showValue:true,position:'outEnd',numberFormatCode:item.number_format,
          textStyle:{fontSize:32,typeface:font,bold:true,fill:palette.ink}},
      });
      applyPresentationChartFont(chart,{fontFamily:font});
      chartSlides.push(number);
      text(slide,item.side_title,1120,225,420,80,44,palette.blue,true,'chart-interpretation-title');
      text(slide,item.side_text,1120,340,420,340,32,palette.muted,false,'chart-interpretation');
    } else if (item.kind==='image_text') {
      text(slide,item.side_title,55,225,545,90,45,palette.blue,true,'image-interpretation-title');
      text(slide,item.side_text,55,355,545,335,33,palette.muted,false,'image-interpretation');
      await image(slide,item.image,{left:650,top:180,width:900,height:590});
    } else if (item.kind==='image') {
      await image(slide,item.image,{left:50,top:160,width:1500,height:610});
    } else if (item.kind==='candidate_image') {
      for (const [j,label] of ['WT重复1','WT重复2','重复1减重复2'].entries()) {
        text(slide,label,130+500*j,190,420,55,34,palette.blue,true,`image-column-${j}`);
      }
      await image(slide,item.image,{left:50,top:255,width:1500,height:445.5},item.crop);
      text(slide,item.body,55,725,1480,60,31,palette.muted,false,'image-statistics');
    } else if (item.kind==='formula') {
      text(slide,item.formula,80,205,1440,125,61,palette.blue,true,'scale-formula');
      table(slide,item,number,390,310);
    } else throw new Error(`Unsupported slide kind: ${item.kind}`);
    if (item.caption) text(slide,item.caption,55,805,1400,52,27,palette.muted,false,'caption');
  }
  slide.speakerNotes.text=`${item.notes}\n\n资料来源：\n${item.sources.map(sourceLink).join('\n')}`;
  console.log(`Authored ${item.id}: ${item.title.replaceAll('\n',' / ')}`);
}
const candidate=path.join(build,`candidate-${path.basename(finalName)}.pptx`);
await (await PresentationFile.exportPptx(p)).save(candidate);
const result=await finalizePresentation({
  workspaceDir:repo,candidatePath:candidate,finalPath:final,pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-heading-fit'],
  explicitTotalSlideCount:24,requiredNativeTableOwnerSlides:tableSlides,
  requiredNativeChartOwnerSlides:chartSlides,materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
  receiptPath:path.join(build,`${finalName}.validation.json`),
});
const verified=await PresentationFile.importPptx(await FileBlob.load(final));
const previews=path.join(build,`previews-${path.basename(finalName,'.pptx')}`);
await fs.mkdir(previews,{recursive:true});
for (let index=0;index<verified.slides.items.length;index++) {
  const slide=verified.slides.items[index];
  const blob=await verified.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(previews,`slide-${String(index+1).padStart(2,'0')}.png`),
    new Uint8Array(await blob.arrayBuffer()));
  console.log(`Rendered final slide ${index+1}/24`);
}
const preview=await verified.export({format:'webp',montage:true,scale:.2});
await fs.writeFile(path.join(previews,'montage.webp'),new Uint8Array(await preview.arrayBuffer()));
await fs.writeFile(path.join(build,'source-hashes.json'),JSON.stringify(hashes,null,2));
await fs.writeFile(path.join(build,'build-summary.json'),JSON.stringify({
  final,previews,slideCount:24,mainSeconds:500,font,tableSlides,chartSlides,
  finalizer:result,evidenceRevision:spec.evidence_revision,
},null,2));
console.log(`Final presentation ready for visual review: ${final}`);
