program sdltest;
{$MODE OBJFPC}
{$m+}
{$linkframework Cocoa}
{$linkframework OpenGL}
{$linklib SDLmain}
{$linklib SDL}
{$linklib SDL_image}


uses sdl, math, sdl_image, sdl_ttf;

{ # GLOBAL DEFINITIONS }
{ ## GLOBAL CLASSES }

{ ### Obstacles }
type
  coords = record
    x : integer;
    y : integer;
  end;
  coordsPtr = ^coords;
  brainPtr = ^brain;

  realArray = array of real;

  Neuron = Class(TObject)
  private
    bias: real;
    weights: array of real;
    function activationFunction(v : real): real;

  public
    constructor create(outNeurons : integer);
    function output(v : real) : realArray;
    procedure setWeightAndBias(newWeights: realArray; newBias : real);
    procedure naturalMutation();
    procedure printWeights();
  end;

  layer = Array of Neuron;
  perceptronType = array of layer;

  Brain = Class(TObject)
  private
    perceptron: Array of layer;

  public
    constructor create();
    procedure mutate();
    procedure printPerceptron();
    function decide(features: array of real) : boolean;
    function getPerceptron() : perceptronType;
    procedure setNeuronTo(layerIx, neuronIx : integer; newNeuron : neuron);
  end;

  Obstacle = Class(TObject)
  private
    x, y1, y2: integer;
    sprites: array[0..1] of TSDL_Rect;
    blitRects: array[0..1] of TSDL_Rect;

  public
    constructor create(ix : integer);
    function update() : integer; {Returns X position of obstacles}
    function testCollision(birdY: integer) : boolean;
    function getSpriteAddress(ix : integer) : PSDL_Rect;
    function getBlitRectAddress(ix : integer) : PSDL_Rect;
    function getTop() : real;

    procedure reset(ix : integer);
  end;

  { ### Bird }
  Bird = Class(TObject)
  private
    y, speedY, accelY, x, score: integer;
    alive, ranked: boolean;
    sprite, blitRect: TSDL_Rect;
    myBrain: Brain;

  public
    ran : integer;
    constructor create();
    procedure jump();
    procedure createBrain();
    procedure mutate();
    procedure cross(brainClone : perceptronType);
    procedure printPerceptron();
    procedure rank();
    function update(inputs : array of real) : integer; { Returns y position of the bird }
    function die(): integer; { Returns final score of bird }
      { GETTERS }
    function isAlive() : boolean;
    function getCoordinates() : coordsPtr;
    function getSprite(): TSDL_Rect;
    function getSpriteAddress(): PSDL_Rect;
    function getBlitRectAddress(): PSDL_Rect;
    function reset() : integer; { Returns score }
    function getScore(): integer;
    function getBrain() : perceptronType;
    function isRanked() : boolean;
  end;
  TArrObstacles = array[0..10] of Obstacle;



{ ## GLOBAL CONSTANTS }
const
  birdX: integer = 50; { Position de l'oiseau }
  birdY0: integer = 250; { Position de départ de l'oiseau }
  obstacleWidth: integer = 100; { Largeur de l'obstacle }
  obstacleSpace: integer = 200; { Intervalle Y entre la partie haute et basse d'un même obstacle }
  obstacleStep: integer = 10; { Facteur d'incrémentation de la position à chaque boucle }
  obstacleInterval : integer = 550; { Distance avant le prochain obstacle }
  gravity: integer = 2; { Facteur de gravité }
  birdWidth: integer = 50; { Taille de l'oiseau }
  { V2 }
  layersTotal: integer = 3;
  layers : array[0..2] of integer = (3,5,1);
  mutationProbability: real = 0.5;
  ellitism : real = 0.15; { Birds ratio that remain unchanged }
  mutationRange: real = 0.1;
  populationTotal: integer = 100;
  crossoverRate : real = 0.15;
  randomBehavior : real = 0.2;
  taillepolice : integer = 50;

{ ## GLOBAL VARIABLES }
var
  i : integer; { Inter }
  j : integer;
  k : integer;
  ii : integer;
  ij : integer;
  ik : integer;
  iii : integer;
  iij : integer;
  iik : integer;
  tmpMax : integer;
  sdlWindow1: PSDL_Surface;
  sdlEvent: PSDL_Event;
  exitloop: boolean;
  obstacles : TArrObstacles;
  birds : array of Bird;
  total: real; { Used in perceptron }
  interPerceptron : Array of Array of Array of real; // LAYERS OF NEURONS' SYNAPSE WEIGHTS
  distanceToNextObstacle: integer = 1000;
  topOfNextObstacle: real = 0;
  potentialObstacleMemory: integer;
  populationRemaining : integer = -1;
  features : array of real; { FINAL ARRAY OF INPUTS }
  ranking : array of bird;
  memCoords: coordsPtr;
  imageBird, blitImage, bestBird, imageButton, imageSelected : PSDL_Surface;
  imageObstacle : PSDL_Surface;
  state : String;
  choice : Integer;
  police : PTTF_Font;

function rand(min, max : real) : real;
begin
  rand := Random * (max - min) + min;
end;

(*  -------------------------------- *)

{ # CONTENT OF CLASSES }


{ ## NEURON }
constructor Neuron.create(outNeurons : integer);
begin
  { BETWEEN -1 AND 1}
  SetLength(weights, outNeurons);
  for iij := 0 to outNeurons - 1 do
  begin
    weights[iij] := rand(-0.1, 0.1);
  end;

  bias := rand(-0.1, 0.1);
end;


function Neuron.activationFunction(v : real) : real;
begin
  activationFunction := tanh(v);
end;

function Neuron.output(v : real) : realArray;
begin
  SetLength(output, length(weights));

  for iii := 0 to length(output) - 1 do
  begin
    output[iii] := activationFunction(v * weights[iii] + bias);
  end;
end;

procedure Neuron.setWeightAndBias(newWeights : realArray; newBias : real);
begin
  weights := newWeights;
  bias := newBias;
end;

procedure Neuron.naturalMutation();
begin
  for iii := 0 to length(weights) - 1 do
  begin
    if Random > mutationProbability then
    begin
      weights[iii] := weights[iii] + rand(-mutationRange, mutationRange);
    end;
  end;

  if Random > mutationProbability then
  begin
    bias := bias + rand(-mutationRange, mutationRange);
  end;
end;

procedure Neuron.printWeights();
begin
  for ik := 0 to length(weights) - 1 do
  begin
    write(ceil(weights[ik]*100), ' ');
  end;
end;

(*  -------------------------------- *)
constructor Brain.create();
begin
  SetLength(perceptron, layersTotal);

  for ij := 0 to layersTotal - 2 do
  begin
    SetLength(perceptron[ij], layers[ij]);

    for ii := 0 to layers[ij] - 1 do
    begin
      perceptron[ij][ii] := Neuron.create(layers[ij + 1]);
    end;
  end;

  SetLength(perceptron[layersTotal - 1], 1);
  perceptron[layersTotal - 1][0] := Neuron.create(1);
end;

function Brain.getPerceptron() : perceptronType;
begin
  getPerceptron := perceptron;
end;

function Brain.decide(features : array of real) : boolean;
begin
  SetLength(interPerceptron, layersTotal); { FOR NON WEIGHTED INPUTS }

  for k := 0 to layersTotal - 2 do { ALL EXCEPT LAST ONE }
  begin
    SetLength(interPerceptron[k], layers[k]);

    for iii := 0 to layers[k] - 1 do
    begin
      SetLength(interPerceptron[k][iii], layers[k+1]);
    end;
  end;

  SetLength(interPerceptron[layersTotal-1], 1);
  SetLength(interPerceptron[layersTotal-1][0], 1);

  { SET FEATURES IN INTERPERCEPTRON }
  for iik := 0 to layers[0] - 1 do
  begin
    interPerceptron[0][iik] := perceptron[0][iik].output(features[iik]);
  end;

  for ii := 1 to layersTotal - 1 do
  begin
    for ik := 0 to layers[ii] - 1 do { FOR EACH NEURON }
    begin
      total := 0; { SUM OF WEIGHTED SYNAPSES }

      for ij := 0 to layers[ii - 1] - 1 do
      begin
        total := total + interPerceptron[ii - 1][ij][ik]; { !!!! }
        {write(ii,ij,ik, '.',round(interPerceptron[ii - 1][ij][ik]*1000),' ');}
      end;
      interperceptron[ii][ik] := perceptron[ii][ik].output(total);
    end;
  end;

  decide := interperceptron[layersTotal - 1][0][0] > 0;
end;

procedure Brain.setNeuronTo(layerIx, neuronIx: integer; newNeuron: Neuron);
begin
  perceptron[layerIx][neuronIx].bias := newNeuron.bias; { UNREFERENCE THE ARRAY COPY }

  for iij := 0 to length(perceptron[layerIx][neuronIx].weights) - 1 do
  begin
    perceptron[layerIx][neuronIx].weights[iij] := newNeuron.weights[iij];
  end;
end;

procedure Brain.mutate();
begin
  for ii := 0 to layersTotal - 1 do
  begin
    for ij := 0 to layers[ii] - 1 do
    begin
      perceptron[ii][ij].naturalMutation();
    end;
  end;
end;

procedure Brain.printPerceptron();
begin
  for ii := 0 to layersTotal - 1 do
  begin
    writeln(';');
    write(ii, ':');
    for ij := 0 to layers[ii] - 1 do
    begin
      perceptron[ii][ij].printWeights();
      write('B', ceil(perceptron[ii][ij].bias * 1000), ' / ');
    end;
  end;
end;

(*  -------------------------------- *)

{ ## OBSTACLES }
constructor Bird.create();
begin
  score := 0;

  reset();

  accelY := gravity;

  sprite.w := birdWidth;
  sprite.h := birdWidth;
  sprite.x := x;
  sprite.y := y;

  blitRect.w := birdWidth;
  blitRect.h := birdWidth;

  ran := Random(10000);

  createBrain();
end;

procedure Bird.createBrain();
begin
  MyBrain := brain.create();
end;

procedure Bird.jump();
begin
  speedY := - 25;
end;

function Bird.update(inputs : array of real) : integer;
begin
  speedY := speedY + gravity;
  y := y + speedY;
  score := score + 1;

  { DECISION }
  SetLength(features, layers[0]);

  features[0] := y / 1000;
  for j := 1 to layers[0] - 1 do
  begin
    features[j] := inputs[j - 1];
  end;

  // features[2] := (y / 1000) - features[2];

  if myBrain.decide(features) and (choice = 1) then jump();

  if (y > 500 - birdWidth) or (y < 0) then die();
  sprite.y := y;


  update := y;
end;

function Bird.die() : integer;
begin
  if alive then
  begin
    alive := false;
    populationRemaining := populationRemaining - 1;
    die := score;
  end;
end;

function Bird.reset() : integer;
begin
  alive := true;
  ranked := false;

  x := birdX;
  y := birdY0;
  speedY := 0;

  reset := score;

  score := 0;

end;

procedure Bird.mutate();
begin
  myBrain.mutate();
end;

procedure Bird.rank();
begin
  ranked := true;
end;

{ GETTERS }
function Bird.isAlive() : boolean;
begin
  isAlive := alive;
end;

function Bird.isRanked() : boolean;
begin
  isRanked := ranked;
end;

var
  ret : coordsPtr;
function Bird.getCoordinates() : coordsPtr;
begin
  new(ret);
  ret^.x := x;
  ret^.y := y;
  getCoordinates := ret;
end;

function Bird.getSprite() : TSDL_Rect;
begin
  getSprite := sprite;
end;

function Bird.getSpriteAddress() : PSDL_Rect;
begin
  getSpriteAddress := @sprite;
end;

function Bird.getBlitRectAddress() : PSDL_Rect;
begin
  getBlitRectAddress := @blitRect;
end;

function Bird.getScore() : Integer;
begin
  getScore := score;
end;

function Bird.getBrain() : perceptronType;
begin
  getBrain := myBrain.getPerceptron();
end;

procedure Bird.printPerceptron();
begin
  myBrain.printPerceptron();
end;

procedure Bird.cross(brainClone : perceptronType);
begin
  for ii := 0 to layersTotal - 1 do
  begin
    for ij := 0 to layers[ii] - 1 do
    begin
      if Random > crossoverRate then
      begin
        myBrain.setNeuronTo(ii, ij, brainClone[ii][ij]);
      end;
    end;
  end;
end;

{ ## OBSTACLES }
constructor Obstacle.create(ix : integer);
begin
  reset(ix);
end;

function Obstacle.testCollision(birdY : integer) : boolean;
begin
  if ((x >= birdX + birdWidth) or (x + obstacleWidth <= birdX)) then exit(false);

  { => obstacle is on the x-axis of the bird }

  if (birdY > y1) and (birdY + birdWidth < y2) then testCollision := false
  else testCollision := true;
end;

function Obstacle.update() : integer;
begin
  x := x - obstacleStep;

  if (x < -obstacleWidth) then reset(9);

  sprites[0].x := x;
  sprites[1].x := x;

  update := x;
end;

function Obstacle.getBlitRectAddress(ix: Integer) : PSDL_Rect;
begin
  getBlitRectAddress := @blitRects[ix];
end;

function Obstacle.getSpriteAddress(ix: integer) : PSDL_Rect;
begin
  getSpriteAddress := @sprites[ix];
end;

function Obstacle.getTop() : real;
begin
  getTop := sprites[0].h;
end;


procedure Obstacle.reset(ix: integer);
begin
  x := (1 + ix) * obstacleInterval;
  // y1 := 400;
  y1 := ceil(rand(100,300));
  y2 := y1 + obstacleSpace;

  sprites[0].x := x;
  sprites[0].y := 0;
  sprites[0].w := obstacleWidth; { IGNORED }
  sprites[0].h := y1; { IGNORED }

  sprites[1].x := x;
  sprites[1].y := y2;
  sprites[1].w := obstacleWidth; { IGNORED }
  sprites[1].h := 1000; { IGNORED }

  blitRects[0].w := obstacleWidth;
  blitRects[0].h := y1;

  blitRects[1].w := obstacleWidth;
  blitRects[1].h := 1000;
end;

procedure showMenu();
var
  buttons : array[0..2] of TSDL_Rect;
  textPos : array[0..2] of TSDL_Rect;
  imageCurrButton: PSDL_Surface;
  policecolor: PSDL_Color;
  texte : array[0..2] of PSDL_Surface;
  const txt : array[0..2] of String = ('PLAY', 'WATCH', 'QUIT');
begin
  new(policecolor);
  policecolor^.r:=255;
  policecolor^.g:=255;
  policecolor^.b:=255;

  for i := 0 to 2 do
  begin
    buttons[i].w := 400;
    buttons[i].h := 100;
    buttons[i].x := 50;
    buttons[i].y := i * 110 + 110;

    texte[i] := TTF_RENDERTEXT_BLENDED ( police , @txt[i], policecolor^);


    textPos[i].w := 400;
    textPos[i].h := 100;
    textPos[i].x := 200-Length(txt[i])*5;
    textPos[i].y := (i+1)*110 + 20;

    imageCurrButton := imageButton;
    if choice = i then imageCurrButton := imageSelected;

    SDL_BlitSurface(imageCurrButton, nil, sdlWindow1, @buttons[i]);
    SDL_BlitSurface( texte[i] , NIL , sdlWindow1 ,  @textPos[i] );
    SDL_FreeSurface( texte[i] );
  end;
  dispose(policecolor)
end;

{ # Beginning of program }
var
  scoreTxt : PSDL_Surface;
  scorePos : TSDL_RECT;
  scoreColor : PSDL_Color;
  currentScore : Integer;
  scoreStr : String;
  imageBG : PSDL_Surface;
begin
  exitloop := false;
  state := 'menu';
  choice := 0;
  randomize;

  if SDL_Init( SDL_INIT_VIDEO ) < 0 then HALT;
  if TTF_INIT = -1 then HALT;
  //initilization of video subsystem

  imageBird := IMG_Load('./res/shark.png');
  bestBird := IMG_Load('./res/best_shark.png');
  imageObstacle := IMG_Load('./res/obstacle.png');
  imageBG := IMG_Load('./res/bg.png');
  imageSelected := IMG_Load('./res/selected.png');
  imageButton := IMG_Load('./res/button.png');
  police := TTF_OPENFONT ('./res/Vogue.ttf', taillepolice );

  scorePos.w := 0;
  scorePos.h := 0;
  scorePos.x := 240;
  scorePos.y := 50;

  new(scoreColor);
  scoreColor^.r:=255;
  scoreColor^.g:=203;
  scoreColor^.b:=0;

  SetLength(birds, populationTotal);
  for i := 0 to populationTotal - 1 do
  begin
    birds[i] := Bird.create();
  end;

  for i := 0 to 9 do
  begin
    obstacles[i] := Obstacle.create(i);
  end;


  sdlWindow1 := SDL_SetVideoMode(500, 500, 32, SDL_SWSURFACE);
  if sdlWindow1 = nil then HALT;

  new( sdlEvent );

  while exitloop = false do
  begin
    SDL_Flip(sdlWindow1);
    SDL_Delay( 30 );
    while SDL_PollEvent( sdlEvent ) = 1 do
    begin
      case sdlEvent^.type_ of

        //keyboard events
        SDL_KEYDOWN: begin
          if sdlEvent^.key.keysym.sym = SDLK_SPACE then ranking[0].jump();
          if sdlEvent^.key.keysym.sym = SDLK_ESCAPE then state := 'menu';
          if sdlEvent^.key.keysym.sym = SDLK_RETURN then
          begin
            state := 'playing';
            populationRemaining := -1; { RESET }
          end;
          if state = 'menu' then
          begin
            if (sdlEvent^.key.keysym.sym = SDLK_DOWN) and (choice < 2) then choice := choice + 1;
            if (sdlEvent^.key.keysym.sym = SDLK_UP) and (choice > 0) then choice := choice - 1;
          end;
        end;
      end;
    end;

    SDL_BlitSurface(imageBG, NIL, sdlWindow1, NIL);

    if(state = 'menu') then
    begin
      showMenu();
      continue;
    end
    else if (choice = 2) then
    begin
      exitloop := true;
      write('bye');
      break;
    end;


    if (populationRemaining <= 0) then
    begin
      SetLength(ranking, populationTotal);
      distanceToNextObstacle := 1000;
      currentScore := 0;
      populationRemaining := populationTotal;

      {Sort birds}
      for iii := 0 to populationTotal - 1 do
      begin
        { FIND MAX }
        tmpMax := -1;
        for iij := 0 to populationTotal - 1 do
        begin
          if birds[iij].isRanked() then continue;
          tmpMax := Max(tmpMax, birds[iij].getScore());
        end;

        for iij := 0 to populationTotal - 1 do
        begin
          if birds[iij].isRanked() then continue;

          if birds[iij].getScore() = tmpMax then
          begin
            ranking[iii] := birds[iij];
            birds[iij].rank();
            break;
          end;
        end;
      end;

      {ONCE RANKING DONE...}
      writeln(ranking[0].getScore());
      for k := 0 to populationTotal - 1 do
      begin
        if k / populationTotal > ellitism then
        begin
          if k / populationTotal > 1 - randomBehavior then
          begin;
            ranking[k].createBrain();
            ranking[k].cross(ranking[0].getBrain());
          end;

          ranking[k].mutate();
        end;

        ranking[k].reset();
      end;

      for i := 0 to 9 do
      begin
        obstacles[i].reset(i);
      end;
    end;


    for i := 0 to 9 do
    begin
      potentialObstacleMemory := obstacles[i].update();

      if (distanceToNextObstacle < 50) then distanceToNextObstacle := 1000; { Reset to furthest }

      distanceToNextObstacle := Min(potentialObstacleMemory, distanceToNextObstacle);
      if distanceToNextObstacle = potentialObstacleMemory then
      begin
        topOfNextObstacle := obstacles[i].getTop();
      end;
      SDL_BlitSurface(imageObstacle, obstacles[i].getBlitRectAddress(0), sdlWindow1, obstacles[i].getSpriteAddress(0)); { TODO }
      SDL_BlitSurface(imageObstacle, obstacles[i].getBlitRectAddress(1), sdlWindow1, obstacles[i].getSpriteAddress(1)); { TODO }
    end;

    i := populationTotal;
    if (choice = 0) then i := 1; { ONLY FIRST BIRD }
    while i > 0 do
    begin
      i := i - 1;

      if (ranking[i].isAlive() = false) then
      begin
        if (choice = 0) then
        begin
          SDL_Delay(3000);
          state := 'menu';
          break;
        end;

        {ELSE}
        continue;
      end;

      currentScore := floor(max(currentScore, ranking[i].getScore() / 2500));

      for k := 0 to 9 do
      begin
          memCoords := ranking[i].getCoordinates();
          if obstacles[k].testCollision(memCoords^.y) then
          begin
            ranking[i].die();
          end;
      end;

      ranking[i].update([distanceToNextObstacle/ 1000, topOfNextObstacle / 1000 ]);

      blitImage := imageBird;
      if (i = 0) then blitImage := bestBird;

      SDL_BlitSurface(blitImage, ranking[i].getBlitRectAddress(), sdlWindow1, ranking[i].getSpriteAddress()); { TODO }
    end;

    str(currentScore, scoreStr);
    if (currentScore < 10) then scoreStr := '0' + scoreStr;
    scoreTxt := TTF_RENDERTEXT_BLENDED (police , @scoreStr, scoreColor^);
    SDL_BlitSurface( scoreTxt , NIL , sdlWindow1 ,  @scorePos );
    SDL_FreeSurface(scoreTxt);
  end;

  dispose( sdlEvent );
  SDL_FreeSurface( sdlWindow1 );
  SDL_FreeSurface( imageButton );
  SDL_FreeSurface( bestBird );
  SDL_FreeSurface( imageSelected );
  SDL_FreeSurface( imageObstacle);
  SDL_FreeSurface( imageBg );
  TTF_CloseFont ( police );
  TTF_QUIT();

  {shutting down video subsystem}
  SDL_Quit();
end.
